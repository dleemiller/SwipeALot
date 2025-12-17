"""Loss functions for swipe keyboard training."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from swipealot.utils import extract_character_logits


class SwipeLoss(nn.Module):
    """Combined loss for character and path prediction."""

    def __init__(
        self,
        char_weight: float = 1.0,
        path_weight: float = 0.1,
        length_weight: float = 0.0,
        focal_gamma: float = 0.0,
        char_class_weights: torch.Tensor | None = None,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        matryoshka_dims: list[int] | None = None,
        matryoshka_weights: list[float] | None = None,
    ):
        """
        Initialize loss function.

        Args:
            char_weight: Weight for character prediction loss
            path_weight: Weight for path prediction loss
        """
        super().__init__()
        self.char_weight = char_weight
        self.path_weight = path_weight
        self.length_weight = length_weight
        self.focal_gamma = focal_gamma
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        # Matryoshka settings
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights
        if self.matryoshka_dims is not None:
            if any(d <= 0 for d in self.matryoshka_dims):
                raise ValueError("Matryoshka dims must be positive")

            if self.matryoshka_weights is not None and len(self.matryoshka_weights) != len(
                self.matryoshka_dims
            ):
                raise ValueError("matryoshka_weights must match matryoshka_dims")

        if char_class_weights is not None:
            # Register for device transfers and checkpointing
            self.register_buffer("char_class_weights", char_class_weights.float())
        else:
            self.char_class_weights = None
        self.path_loss_fn = nn.MSELoss(reduction="none")

    def _infonce_from_embeddings(
        self,
        emb: torch.Tensor,  # [N, d]
        pair_ids: torch.Tensor,  # [N]
        temperature: float,
        gradient_mask: torch.Tensor | None,  # [N] 1=query, 0=key
    ) -> torch.Tensor:
        # Apply gradient masking: detach keys, keep queries
        if gradient_mask is not None:
            emb = torch.where(gradient_mask.unsqueeze(1).bool(), emb, emb.detach())
        else:
            emb = emb.detach()

        emb = torch.nn.functional.normalize(emb, dim=-1)

        sim = (emb @ emb.transpose(0, 1)) / temperature
        sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9

        pos_mask = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
        pos_mask.fill_diagonal_(False)

        if gradient_mask is not None:
            query_indices = (gradient_mask == 1).nonzero(as_tuple=True)[0]
        else:
            query_indices = torch.arange(sim.size(0), device=sim.device)

        if query_indices.numel() == 0:
            return torch.tensor(0.0, device=sim.device)

        query_pos_mask = pos_mask[query_indices]
        has_positive = query_pos_mask.any(dim=1)
        if not has_positive.any():
            return torch.tensor(0.0, device=sim.device)

        valid_query_indices = query_indices[has_positive]
        valid_pos_mask = query_pos_mask[has_positive]

        # first positive per query
        first_pos_idx = valid_pos_mask.byte().argmax(dim=1)
        pos_sims = sim[valid_query_indices, first_pos_idx]
        logsumexp = torch.logsumexp(sim[valid_query_indices], dim=1)

        return -(pos_sims - logsumexp).mean()

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            outputs: Model outputs with 'char_logits' and optionally 'path_coords_pred'
            batch: Batch data with labels

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Character prediction loss
        # Handle both dict and dataclass outputs
        if isinstance(outputs, dict):
            char_logits = outputs.get("char_logits")  # [batch, full_seq_len, vocab_size] or None
        else:
            char_logits = getattr(outputs, "char_logits", None)
        char_labels = batch.get("char_labels")

        if char_logits is None or char_labels is None:
            # Skip char loss if head disabled or labels absent
            device = batch["path_coords"].device
            char_loss = torch.tensor(0.0, device=device)
        else:
            # Extract character portion from sequence
            # Sequence structure: [CLS] + path + [SEP] + chars
            path_len = batch["path_coords"].shape[1]
            char_logits_subset = extract_character_logits(
                char_logits, path_len, char_labels.shape[1]
            )

            # Flatten for loss computation
            char_logits_flat = char_logits_subset.reshape(-1, char_logits_subset.shape[-1])
            char_labels_flat = char_labels.reshape(-1)

            # Only keep supervised positions
            mask = char_labels_flat != -100
            if mask.any():
                logits_supervised = char_logits_flat[mask]
                labels_supervised = char_labels_flat[mask]

                # Use F.cross_entropy for cleaner implementation
                loss_terms = F.cross_entropy(
                    logits_supervised, labels_supervised, reduction="none", weight=None
                )

                # Optional class frequency weighting
                if self.char_class_weights is not None:
                    loss_terms = loss_terms * self.char_class_weights[labels_supervised]

                # Optional focal modulation (focal loss formulation)
                if self.focal_gamma > 0.0:
                    # Compute pt (probability of true class) for focal weighting
                    pt = torch.exp(-loss_terms)  # Since loss = -log(pt), pt = exp(-loss)
                    focal_weight = (1 - pt) ** self.focal_gamma
                    loss_terms = focal_weight * loss_terms

                char_loss = loss_terms.mean()
            else:
                char_loss = torch.tensor(0.0, device=char_logits.device)
        losses["char_loss"] = char_loss

        # Path prediction loss (if enabled)
        has_path_pred = (
            "path_coords_pred" in outputs
            if isinstance(outputs, dict)
            else hasattr(outputs, "path_coords_pred")
        )
        if has_path_pred and batch.get("path_labels") is not None:
            if isinstance(outputs, dict):
                path_pred = outputs["path_coords_pred"]  # [batch, full_seq_len, 3]
            else:
                path_pred = outputs.path_coords_pred  # [batch, full_seq_len, 3]
            path_labels = batch["path_labels"]  # [batch, path_len, 3]
            path_mask_indices = batch["path_mask_indices"]  # [batch, path_len]

            # Extract path portion from sequence
            path_start = 1  # Skip [CLS]
            path_end = 1 + path_len
            path_pred_subset = path_pred[:, path_start:path_end, :]  # [batch, path_len, 3]

            # Compute MSE only on masked positions
            path_loss = self.path_loss_fn(path_pred_subset, path_labels)  # [batch, path_len, 3]

            # Apply mask: only compute loss where we actually masked points
            path_mask_expanded = path_mask_indices.unsqueeze(-1).float()  # [batch, path_len, 1]
            path_loss = (path_loss * path_mask_expanded).sum()

            # Normalize by number of masked points
            num_masked = path_mask_indices.sum()
            if num_masked > 0:
                path_loss = path_loss / num_masked
            else:
                path_loss = torch.tensor(0.0, device=path_loss.device)

            losses["path_loss"] = path_loss

        # Total loss
        total_loss = self.char_weight * char_loss
        if "path_loss" in losses:
            total_loss = total_loss + self.path_weight * losses["path_loss"]

        # CLS length prediction (optional)
        has_length_logits = (
            "length_logits" in outputs
            if isinstance(outputs, dict)
            else hasattr(outputs, "length_logits")
        )
        if (
            self.length_weight > 0.0
            and has_length_logits
            and "length_target" in batch
            and "length_supervise_mask" in batch
        ):
            if isinstance(outputs, dict):
                length_pred = outputs["length_logits"]  # [batch]
            else:
                length_pred = outputs.length_logits  # [batch]
            length_target = batch["length_target"].float()  # [batch]
            supervise_mask = batch["length_supervise_mask"].bool()  # [batch]
            if supervise_mask.any():
                huber = nn.SmoothL1Loss(reduction="none")
                length_loss_terms = huber(length_pred, length_target)
                length_loss = length_loss_terms[supervise_mask].mean()
            else:
                length_loss = torch.tensor(0.0, device=length_pred.device)
            losses["length_loss"] = length_loss
            total_loss = total_loss + self.length_weight * length_loss

        # Contrastive (Matryoshka-capable)
        if self.contrastive_weight > 0.0 and "pair_ids" in batch:
            # Use last_hidden_state (always available) instead of hidden_states (only with output_hidden_states=True)
            if isinstance(outputs, dict):
                hidden_states = outputs.get(
                    "last_hidden_state", outputs.get("hidden_states")
                )  # [N, seq_len, d]
            else:
                hidden_states = outputs.last_hidden_state  # [N, seq_len, d]
            pair_ids = batch["pair_ids"]  # [N]
            gradient_mask = batch.get("gradient_mask")  # [N]
            path_len = batch["path_coords"].shape[1]

            sep_position = 1 + path_len
            sep_embeddings = hidden_states[:, sep_position, :]  # [N, d]
            d = sep_embeddings.size(-1)

            # Decide which dims to use
            if self.matryoshka_dims is None:
                dims = [d]  # original behavior
            else:
                dims = [int(x) for x in self.matryoshka_dims]
                # keep valid, unique, sorted, and ensure full dim included if you want
                dims = sorted({x for x in dims if 1 <= x <= d})
                if len(dims) == 0:
                    dims = [d]

            # Weights
            if self.matryoshka_weights is None:
                weights = [1.0] * len(dims)
            else:
                if len(self.matryoshka_weights) != len(dims):
                    raise ValueError("matryoshka_weights must match matryoshka_dims length")
                weights = [float(w) for w in self.matryoshka_weights]

            # Compute per-dim InfoNCE and combine
            per_dim_losses = []
            for dim in dims:
                emb_d = sep_embeddings[:, :dim]
                per_dim_losses.append(
                    self._infonce_from_embeddings(
                        emb=emb_d,
                        pair_ids=pair_ids,
                        temperature=self.contrastive_temperature,
                        gradient_mask=gradient_mask,
                    )
                )

            wsum = sum(weights)
            contrastive_loss = sum(w * L for w, L in zip(weights, per_dim_losses, strict=False)) / (
                wsum if wsum > 0 else 1.0
            )

            losses["contrastive_loss"] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        losses["total_loss"] = total_loss
        return losses


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss for cross-encoder training.

    Used for training with (anchor, positive, negative_1, ..., negative_n) tuples.
    Implemented as CrossEntropyLoss where the positive is always at index 0.

    Based on SBERT's implementation for cross-encoders.
    Reference: https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, scale: float = 10.0, activation_fn: nn.Module | None = None):
        """
        Initialize MNR loss.

        Args:
            scale: Output of similarity function is multiplied by scale value (default: 10.0)
            activation_fn: Activation function applied to logits before scaling (default: Sigmoid)
        """
        super().__init__()
        self.scale = scale
        self.activation_fn = activation_fn if activation_fn is not None else nn.Sigmoid()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, scores: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute MNR loss.

        Args:
            scores: [batch, 1+N] similarity scores
                    First column is positive pair, rest are negatives
            labels: [batch] labels (all zeros, indicating positive is at index 0)
                    If None, will be created automatically

        Returns:
            Scalar loss
        """
        # Apply activation function (SBERT uses Sigmoid by default)
        if self.activation_fn:
            scores = self.activation_fn(scores)

        # Apply temperature scaling
        if self.scale:
            scores = scores * self.scale

        # Create labels if not provided (positive always at index 0)
        if labels is None:
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

        # CrossEntropyLoss: maximize probability of index 0 (positive)
        loss = self.cross_entropy(scores, labels)

        return loss

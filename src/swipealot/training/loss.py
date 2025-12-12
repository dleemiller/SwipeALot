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
        focal_gamma: float = 0.0,
        char_class_weights: torch.Tensor | None = None,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
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
        self.focal_gamma = focal_gamma
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        if char_class_weights is not None:
            # Register for device transfers and checkpointing
            self.register_buffer("char_class_weights", char_class_weights.float())
        else:
            self.char_class_weights = None
        self.path_loss_fn = nn.MSELoss(reduction="none")

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
        char_logits = outputs["char_logits"]  # [batch, full_seq_len, vocab_size]
        char_labels = batch["char_labels"]  # [batch, char_len]

        # Extract character portion from sequence
        # Sequence structure: [CLS] + path + [SEP] + chars
        path_len = batch["path_coords"].shape[1]
        char_logits_subset = extract_character_logits(char_logits, path_len, char_labels.shape[1])

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
        if "path_coords_pred" in outputs and batch.get("path_labels") is not None:
            path_pred = outputs["path_coords_pred"]  # [batch, full_seq_len, 3]
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

        # Contrastive loss (pairwise views using SEP token embeddings)
        if self.contrastive_weight > 0.0 and "pair_ids" in batch:
            hidden_states = outputs["hidden_states"]  # [N, seq_len, d]
            pair_ids = batch["pair_ids"]  # [N]
            gradient_mask = batch.get("gradient_mask")  # [N] - 1 = query, 0 = key
            path_len = batch["path_coords"].shape[1]

            # Extract SEP token embeddings (position = 1 + path_len)
            sep_position = 1 + path_len
            sep_embeddings = hidden_states[:, sep_position, :]  # [N, d]

            # Apply gradient masking: detach keys, keep queries
            if gradient_mask is not None:
                # Create masked embeddings: queries keep gradients, keys are detached
                sep_embeddings_masked = torch.where(
                    gradient_mask.unsqueeze(1).bool(), sep_embeddings, sep_embeddings.detach()
                )
            else:
                # Fallback: detach all (conservative default)
                sep_embeddings_masked = sep_embeddings.detach()

            # Normalize after gradient masking
            sep_embeddings_norm = torch.nn.functional.normalize(sep_embeddings_masked, dim=-1)

            # For similarity computation, we want:
            # - Queries (gradient_mask=1) to compute similarity to all
            # - Keys provide targets but don't get gradients
            # We use the normalized embeddings for both sides
            sim = (
                torch.matmul(sep_embeddings_norm, sep_embeddings_norm.transpose(0, 1))
                / self.contrastive_temperature
            )

            # Exclude self
            sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9

            # Positive mask (same pair_id, not self)
            pos_mask = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
            pos_mask.fill_diagonal_(False)

            # Only compute loss for query samples (gradient_mask=1)
            if gradient_mask is not None:
                query_indices = (gradient_mask == 1).nonzero(as_tuple=True)[0]
            else:
                query_indices = torch.arange(sim.size(0), device=sim.device)

            if query_indices.numel() > 0:
                # Vectorized: get positive mask for queries [num_queries, N]
                query_pos_mask = pos_mask[query_indices]

                # Check which queries have at least one positive
                has_positive = query_pos_mask.any(dim=1)

                if has_positive.any():
                    # Filter to queries with positives
                    valid_query_indices = query_indices[has_positive]
                    valid_pos_mask = query_pos_mask[has_positive]

                    # For each query, get index of first positive
                    first_pos_idx = valid_pos_mask.byte().argmax(dim=1)

                    # Gather positive similarities
                    pos_sims = sim[valid_query_indices, first_pos_idx]

                    # Compute log-sum-exp for these queries
                    logsumexp = torch.logsumexp(sim[valid_query_indices], dim=1)

                    # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
                    contrastive_loss = -(pos_sims - logsumexp).mean()
                else:
                    contrastive_loss = torch.tensor(0.0, device=sim.device)
            else:
                contrastive_loss = torch.tensor(0.0, device=sim.device)

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

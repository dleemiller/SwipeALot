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
        path_loss_dims: list[int] | None = None,
        path_loss_end_weight: float = 1.0,
        path_loss_radial_weight: float = 0.0,
        path_sigma_min: float = 0.1,
        path_best_of_k: int = 1,
        path_sigma_target_min: float = 0.1,
        path_sigma_target_max: float = 0.4,
        uncertainty_reg_weight: float = 0.1,
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
        self.path_loss_dims = path_loss_dims[:] if path_loss_dims is not None else None
        self.path_loss_end_weight = float(path_loss_end_weight)
        self.path_loss_radial_weight = float(path_loss_radial_weight)
        if path_sigma_min <= 0.0:
            raise ValueError("path_sigma_min must be > 0")
        self.path_sigma_min = float(path_sigma_min)
        if path_best_of_k < 1:
            raise ValueError("path_best_of_k must be >= 1")
        self.path_best_of_k = int(path_best_of_k)

        # Uncertainty regularization parameters
        self.path_sigma_target_min = float(path_sigma_target_min)
        self.path_sigma_target_max = float(path_sigma_target_max)
        self.uncertainty_reg_weight = float(uncertainty_reg_weight)

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
        self.path_nll_loss = nn.GaussianNLLLoss(reduction="none", full=True)

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

    @staticmethod
    def _get_output(outputs: object, key: str) -> torch.Tensor | None:
        if isinstance(outputs, dict):
            return outputs.get(key)
        return getattr(outputs, key, None)

    def _compute_char_loss(
        self, *, char_logits: torch.Tensor | None, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        char_labels = batch.get("char_labels")
        if char_logits is None or char_labels is None:
            device = batch["path_coords"].device
            return torch.tensor(0.0, device=device)

        # Support both:
        # - `char_logits` over the text segment only: [B, char_len, V]
        # - legacy `char_logits` over the full mixed sequence: [B, seq_len, V]
        if char_logits.shape[1] == char_labels.shape[1]:
            char_logits_subset = char_logits
        else:
            # Sequence structure: [CLS] + path + [SEP] + chars
            path_len = batch["path_coords"].shape[1]
            char_logits_subset = extract_character_logits(
                char_logits, path_len, char_labels.shape[1]
            )

        char_logits_flat = char_logits_subset.reshape(-1, char_logits_subset.shape[-1])
        char_labels_flat = char_labels.reshape(-1)

        # Only keep supervised positions
        supervised_mask = char_labels_flat != -100
        if not supervised_mask.any():
            return torch.tensor(0.0, device=char_logits.device)

        logits_supervised = char_logits_flat[supervised_mask]
        labels_supervised = char_labels_flat[supervised_mask]

        loss_terms = F.cross_entropy(
            logits_supervised, labels_supervised, reduction="none", weight=None
        )

        if self.char_class_weights is not None:
            loss_terms = loss_terms * self.char_class_weights[labels_supervised]

        if self.focal_gamma > 0.0:
            pt = torch.exp(-loss_terms)  # loss = -log(pt)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss_terms = focal_weight * loss_terms

        return loss_terms.mean()

    def _compute_path_loss(
        self,
        *,
        path_pred: torch.Tensor | None,
        path_log_sigma: torch.Tensor | None,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if path_pred is None or batch.get("path_labels") is None:
            return None

        path_labels = batch["path_labels"]  # [B, path_len, path_input_dim]
        path_mask_indices = batch["path_mask_indices"]  # [B, path_len]
        path_len = path_labels.shape[1]
        path_labels_full = path_labels
        path_log_sigma_subset = None

        # Support both:
        # - `path_pred` over the path segment only: [B, path_len, D]
        # - legacy `path_pred` over the full mixed sequence: [B, seq_len, D]
        if path_pred.shape[1] == path_len:
            path_pred_subset = path_pred
        else:
            path_start = 1  # Skip [CLS]
            path_end = 1 + path_len
            path_pred_subset = path_pred[:, path_start:path_end, :]
        if path_log_sigma is not None:
            if path_log_sigma.shape[1] == path_len:
                path_log_sigma_subset = path_log_sigma
            else:
                path_start = 1  # Skip [CLS]
                path_end = 1 + path_len
                path_log_sigma_subset = path_log_sigma[:, path_start:path_end, :]

        if self.path_loss_dims is not None:
            dims = [int(d) for d in self.path_loss_dims]
            if len(dims) == 0:
                raise ValueError("path_loss_dims must be non-empty when provided")
            max_dim = int(path_labels.shape[-1])
            if any(d < 0 or d >= max_dim for d in dims):
                raise ValueError(f"path_loss_dims {dims} out of range for path_input_dim={max_dim}")
            path_pred_subset = path_pred_subset[..., dims]
            path_labels = path_labels[..., dims]
            if path_log_sigma_subset is not None:
                path_log_sigma_subset = path_log_sigma_subset[..., dims]

        if path_log_sigma_subset is None or self.path_best_of_k <= 1:
            if path_log_sigma_subset is None:
                path_loss = self.path_loss_fn(path_pred_subset, path_labels)  # [B, path_len, D]
            else:
                var = torch.exp(2.0 * path_log_sigma_subset)
                var = torch.clamp(var, min=self.path_sigma_min * self.path_sigma_min)
                path_loss = self.path_nll_loss(path_pred_subset, path_labels, var)
        else:
            sigma = torch.exp(path_log_sigma_subset)
            sigma = torch.clamp(sigma, min=self.path_sigma_min)
            k = self.path_best_of_k
            eps = torch.randn(
                (k,) + tuple(sigma.shape),
                device=sigma.device,
                dtype=sigma.dtype,
            )
            samples = path_pred_subset.unsqueeze(0) + sigma.unsqueeze(0) * eps
            labels = path_labels.unsqueeze(0).to(dtype=samples.dtype)
            path_loss = (samples - labels) ** 2  # [K, B, path_len, D]
        weight = path_mask_indices.unsqueeze(-1).float()  # [B, path_len, 1]

        if self.path_loss_end_weight != 1.0:
            weights = torch.linspace(
                1.0,
                float(self.path_loss_end_weight),
                path_len,
                device=path_loss.device,
                dtype=path_loss.dtype,
            ).view(1, path_len, 1)
            weight = weight * weights

        if self.path_loss_radial_weight != 0.0:
            if int(path_labels_full.shape[-1]) < 2:
                raise ValueError("path_loss_radial_weight requires x/y dims in path_labels")
            center_x, center_y = 0.5, 0.5
            dx = path_labels_full[..., 0] - center_x
            dy = path_labels_full[..., 1] - center_y
            dist = torch.sqrt(dx * dx + dy * dy)
            max_dx = max(center_x, 1.0 - center_x)
            max_dy = max(center_y, 1.0 - center_y)
            max_dist = (max_dx * max_dx + max_dy * max_dy) ** 0.5
            if max_dist > 0:
                dist = dist / max_dist
            radial_weight = (1.0 + self.path_loss_radial_weight * dist).unsqueeze(-1)
            weight = weight * radial_weight

        if path_loss.dim() == 4:
            weighted = path_loss * weight.unsqueeze(0)
            per_sample_sum = weighted.sum(dim=(2, 3))  # [K, B]
            min_sum, _ = per_sample_sum.min(dim=0)  # [B]
            path_loss = min_sum.sum()
        else:
            path_loss = (path_loss * weight).sum()

        denom = weight.sum()
        if denom > 0:
            return path_loss / denom
        return torch.tensor(0.0, device=path_loss.device)

    def _compute_length_loss(
        self, *, length_pred: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        length_target = batch["length_target"].float()  # [B]
        supervise_mask = batch["length_supervise_mask"].bool()  # [B]
        if not supervise_mask.any():
            return torch.tensor(0.0, device=length_pred.device)

        huber = nn.SmoothL1Loss(reduction="none")
        length_loss_terms = huber(length_pred, length_target)
        return length_loss_terms[supervise_mask].mean()

    def _compute_contrastive_loss(
        self, *, hidden_states: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pair_ids = batch["pair_ids"]  # [N]
        gradient_mask = batch.get("gradient_mask")  # [N]
        path_len = batch["path_coords"].shape[1]

        sep_position = 1 + path_len
        sep_embeddings = hidden_states[:, sep_position, :]  # [N, d]
        d = sep_embeddings.size(-1)

        if self.matryoshka_dims is None:
            dims = [d]
        else:
            dims = sorted({int(x) for x in self.matryoshka_dims if 1 <= int(x) <= d})
            if len(dims) == 0:
                dims = [d]

        if self.matryoshka_weights is None:
            weights = [1.0] * len(dims)
        else:
            if len(self.matryoshka_weights) != len(dims):
                raise ValueError("matryoshka_weights must match matryoshka_dims length")
            weights = [float(w) for w in self.matryoshka_weights]

        per_dim_losses: list[torch.Tensor] = []
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
        return sum(w * L for w, L in zip(weights, per_dim_losses, strict=False)) / (
            wsum if wsum > 0 else 1.0
        )

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
        losses: dict[str, torch.Tensor] = {}

        char_logits = self._get_output(outputs, "char_logits")
        char_loss = self._compute_char_loss(char_logits=char_logits, batch=batch)
        losses["char_loss"] = char_loss

        path_pred = self._get_output(outputs, "path_logits")
        if path_pred is None:
            path_pred = self._get_output(outputs, "path_coords_pred")
        path_log_sigma = self._get_output(outputs, "path_log_sigma")
        path_loss = self._compute_path_loss(
            path_pred=path_pred, path_log_sigma=path_log_sigma, batch=batch
        )
        if path_loss is not None:
            losses["path_loss"] = path_loss

            # Diagnostic metrics for center collapse analysis
            if path_pred is not None and "path_labels" in batch:
                path_labels = batch["path_labels"]
                path_len = path_labels.shape[1]

                # Handle both full sequence and path-only predictions
                if path_pred.shape[1] == path_len:
                    path_pred_subset = path_pred
                else:
                    path_start = 1  # Skip [CLS]
                    path_end = 1 + path_len
                    path_pred_subset = path_pred[:, path_start:path_end, :]

                sigma_for_reg = None
                if path_log_sigma is not None:
                    if path_log_sigma.shape[1] == path_len:
                        sigma_for_reg = torch.exp(path_log_sigma[..., :2])  # x,y only
                    else:
                        path_start = 1
                        path_end = 1 + path_len
                        sigma_for_reg = torch.exp(path_log_sigma[:, path_start:path_end, :2])

                if sigma_for_reg is not None and self.uncertainty_reg_weight > 0:
                    full_path_sigma = sigma_for_reg
                    eps = 1e-8
                    log_sigma = torch.log(full_path_sigma + eps)
                    log_sigma_min = torch.log(
                        torch.tensor(self.path_sigma_target_min, device=log_sigma.device)
                    )
                    log_sigma_max = torch.log(
                        torch.tensor(self.path_sigma_target_max, device=log_sigma.device)
                    )
                    too_low_penalty = torch.clamp(log_sigma_min - log_sigma, min=0.0).pow(2)
                    too_high_penalty = torch.clamp(log_sigma - log_sigma_max, min=0.0).pow(2)
                    uncertainty_penalty = (too_low_penalty + too_high_penalty).mean()
                    losses["uncertainty_penalty"] = (
                        uncertainty_penalty * self.uncertainty_reg_weight
                    )

                with torch.no_grad():
                    # Sigma diagnostics by position
                    if sigma_for_reg is not None and self.training:
                        sigma = sigma_for_reg.detach()

                        # Compute by terciles
                        third = path_len // 3
                        start_sigma = sigma[:, :third].mean()
                        mid_sigma = sigma[:, third : 2 * third].mean()
                        end_sigma = sigma[:, -third:].mean()

                        losses["debug/sigma_start"] = start_sigma
                        losses["debug/sigma_mid"] = mid_sigma
                        losses["debug/sigma_end"] = end_sigma
                        losses["debug/sigma_end_to_start_ratio"] = end_sigma / (start_sigma + 1e-8)

                    # Center collapse metrics
                    center = torch.tensor([0.5, 0.5], device=path_pred_subset.device)

                    # Distance from center by position (x, y coords only)
                    third = path_pred_subset.shape[1] // 3
                    start_dist = torch.norm(path_pred_subset[:, :third, :2] - center, dim=-1).mean()
                    end_dist = torch.norm(path_pred_subset[:, -third:, :2] - center, dim=-1).mean()

                    true_end_dist = torch.norm(path_labels[:, -third:, :2] - center, dim=-1).mean()

                    losses["debug/pred_start_center_dist"] = start_dist
                    losses["debug/pred_end_center_dist"] = end_dist
                    losses["debug/true_end_center_dist"] = true_end_dist
                    losses["debug/collapse_ratio"] = end_dist / (true_end_dist + 1e-8)

        # Total loss
        total_loss = self.char_weight * char_loss
        if "path_loss" in losses:
            total_loss = total_loss + self.path_weight * losses["path_loss"]

        # Add uncertainty regularization penalty
        if "uncertainty_penalty" in losses:
            total_loss = total_loss + losses["uncertainty_penalty"]

        # CLS length prediction (optional)
        has_length_logits = self._get_output(outputs, "length_logits") is not None
        if (
            self.length_weight > 0.0
            and has_length_logits
            and "length_target" in batch
            and "length_supervise_mask" in batch
        ):
            length_pred = self._get_output(outputs, "length_logits")
            assert length_pred is not None
            length_loss = self._compute_length_loss(length_pred=length_pred, batch=batch)
            losses["length_loss"] = length_loss
            total_loss = total_loss + self.length_weight * length_loss

        # Contrastive (Matryoshka-capable)
        if self.contrastive_weight > 0.0 and "pair_ids" in batch:
            # Use last_hidden_state (always available) instead of hidden_states (only with output_hidden_states=True)
            hidden_states = self._get_output(outputs, "last_hidden_state")
            if hidden_states is None:
                hidden_states = self._get_output(outputs, "hidden_states")
            if hidden_states is None:
                raise ValueError("contrastive loss enabled but model did not return hidden states")

            contrastive_loss = self._compute_contrastive_loss(
                hidden_states=hidden_states, batch=batch
            )
            losses["contrastive_loss"] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        losses["total_loss"] = total_loss
        return losses


# Legacy note: the cross-encoder `MultipleNegativesRankingLoss` lives in
# `archive/cross_encoder/training/loss.py` (archived cross-encoder stack).

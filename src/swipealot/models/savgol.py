"""Savitzky-Golay preprocessing as a PyTorch nn.Module.

Computes the same 8D features as the NumPy preprocessing in
``swipealot.data.preprocessing.preprocess_raw_path_to_sg_features``,
using frozen Conv1d layers with SG coefficients (window=7, poly=2).

Input:  [B, 2, T]  resampled (x, y) positions
Output: [B, T, 8]  features (x, y, dx, dy, d2x, d2y, speed, curvature)

The output is transposed to [B, T, D] to match SwipeALot's convention
(as opposed to encodercnn's [B, D, T]).
"""

import torch
import torch.nn as nn

# Savitzky-Golay coefficients: window=7, poly=2
SAVGOL_DERIV1 = [0.10714286, 0.07142857, 0.03571429, 0.0, -0.03571429, -0.07142857, -0.10714286]
SAVGOL_DERIV2 = [
    0.11904762,
    0.02380952,
    -0.04761905,
    -0.0952381,
    -0.04761905,
    0.02380952,
    0.11904762,
]


class SavgolPreprocessor(nn.Module):
    """Savitzky-Golay feature computation as a PyTorch module.

    Two frozen ``Conv1d`` layers compute first and second derivatives via
    pre-computed SG coefficients.  Replicate padding preserves sequence
    length and matches the NumPy edge-clamping behaviour.

    Args:
        eps: Small constant added inside ``sqrt`` for numerical stability.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

        # Conv1d: in=1, out=1, kernel=7, no built-in padding (we pad manually)
        self.deriv1_conv = nn.Conv1d(1, 1, kernel_size=7, padding=0, bias=False)
        self.deriv2_conv = nn.Conv1d(1, 1, kernel_size=7, padding=0, bias=False)

        self.deriv1_conv.weight.data = torch.tensor(SAVGOL_DERIV1, dtype=torch.float32).view(
            1, 1, 7
        )
        self.deriv2_conv.weight.data = torch.tensor(SAVGOL_DERIV2, dtype=torch.float32).view(
            1, 1, 7
        )

        self.deriv1_conv.weight.requires_grad = False
        self.deriv2_conv.weight.requires_grad = False

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Compute 8D SG features from (x, y) positions.

        Args:
            xy: ``[B, 2, T]`` tensor — channel 0 is *x*, channel 1 is *y*.

        Returns:
            ``[B, T, 8]`` tensor with channels
            (x, y, dx, dy, d2x, d2y, speed, curvature).
        """
        x = xy[:, 0:1, :]  # [B, 1, T]
        y = xy[:, 1:2, :]

        dx = self._conv_replicate(x, self.deriv1_conv)
        dy = self._conv_replicate(y, self.deriv1_conv)
        d2x = self._conv_replicate(x, self.deriv2_conv)
        d2y = self._conv_replicate(y, self.deriv2_conv)

        speed = torch.sqrt(dx**2 + dy**2 + self.eps)

        theta = torch.atan2(dy, dx)
        theta_unwrapped = self._unwrap(theta)
        curvature = self._conv_replicate(theta_unwrapped, self.deriv1_conv)
        curvature = torch.clamp(curvature, -2.0, 2.0)

        # [B, 8, T] → [B, T, 8]
        out = torch.cat([x, y, dx, dy, d2x, d2y, speed, curvature], dim=1)
        return out.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _conv_replicate(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Conv1d with replicate padding (3 each side for kernel=7)."""
        x_padded = torch.nn.functional.pad(x, (3, 3), mode="replicate")
        return conv(x_padded)

    @staticmethod
    def _unwrap(phase: torch.Tensor) -> torch.Tensor:
        """Unwrap phase angles (equivalent to ``numpy.unwrap``)."""
        diff = phase[:, :, 1:] - phase[:, :, :-1]
        pi = 3.141592653589793
        correction = torch.zeros_like(diff)
        correction = torch.where(diff > pi, correction - 2 * pi, correction)
        correction = torch.where(diff < -pi, correction + 2 * pi, correction)
        cumulative = torch.cumsum(correction, dim=-1)
        unwrapped = phase.clone()
        unwrapped[:, :, 1:] = phase[:, :, 1:] + cumulative
        return unwrapped

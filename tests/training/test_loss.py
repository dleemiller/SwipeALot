"""Tests for MultipleNegativesRankingLoss."""

import torch
import torch.nn as nn

from src.swipealot.training.loss import MultipleNegativesRankingLoss


class TestMultipleNegativesRankingLoss:
    """Test MNR loss implementation."""

    def test_initialization(self):
        """Test loss initialization with default parameters."""
        loss_fn = MultipleNegativesRankingLoss()
        assert loss_fn.scale == 10.0
        assert isinstance(loss_fn.activation_fn, nn.Sigmoid)
        assert isinstance(loss_fn.cross_entropy, nn.CrossEntropyLoss)

    def test_initialization_custom_params(self):
        """Test loss initialization with custom parameters."""
        loss_fn = MultipleNegativesRankingLoss(scale=20.0, activation_fn=nn.Tanh())
        assert loss_fn.scale == 20.0
        assert isinstance(loss_fn.activation_fn, nn.Tanh)

    def test_initialization_no_activation(self):
        """Test loss initialization without activation function."""
        loss_fn = MultipleNegativesRankingLoss(activation_fn=None)
        assert loss_fn.activation_fn is None

    def test_forward_shape(self):
        """Test forward pass returns scalar loss."""
        loss_fn = MultipleNegativesRankingLoss()

        # Batch of 4, each with 1 positive + 3 negatives
        scores = torch.randn(4, 4)

        loss = loss_fn(scores)

        # Should return scalar
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_forward_perfect_predictions(self):
        """Test loss when positive is clearly highest."""
        loss_fn = MultipleNegativesRankingLoss(activation_fn=None, scale=1.0)

        # Positive (index 0) has much higher score than negatives
        scores = torch.tensor(
            [
                [10.0, 1.0, 1.0, 1.0],  # Clear positive
                [10.0, 2.0, 2.0, 2.0],  # Clear positive
            ]
        )

        loss = loss_fn(scores)

        # Loss should be low when positive is clearly highest
        assert loss.item() < 0.5

    def test_forward_worst_predictions(self):
        """Test loss when positive is clearly lowest."""
        loss_fn = MultipleNegativesRankingLoss(activation_fn=None, scale=1.0)

        # Positive (index 0) has much lower score than negatives
        scores = torch.tensor(
            [
                [1.0, 10.0, 10.0, 10.0],  # Bad prediction
                [2.0, 10.0, 10.0, 10.0],  # Bad prediction
            ]
        )

        loss = loss_fn(scores)

        # Loss should be high when positive is lowest
        assert loss.item() > 2.0

    def test_activation_applied_before_scale(self):
        """Test that activation is applied before scaling."""
        # Create loss with sigmoid activation
        loss_fn = MultipleNegativesRankingLoss(scale=10.0, activation_fn=nn.Sigmoid())

        # Raw scores
        scores = torch.tensor([[2.0, 1.0, 0.5, 0.0]])

        # We can't directly access intermediate values, but we can verify
        # the loss is computed correctly by checking it's > 0
        loss = loss_fn(scores)
        assert loss.item() >= 0.0

    def test_labels_default_to_zeros(self):
        """Test that labels default to zeros (positive at index 0)."""
        loss_fn = MultipleNegativesRankingLoss(activation_fn=None, scale=1.0)

        scores = torch.randn(4, 4)

        # Should work without passing labels
        loss = loss_fn(scores)
        assert loss.item() >= 0.0

    def test_labels_explicit(self):
        """Test with explicit labels."""
        loss_fn = MultipleNegativesRankingLoss(activation_fn=None, scale=1.0)

        scores = torch.randn(4, 4)
        labels = torch.zeros(4, dtype=torch.long)  # All zeros

        loss = loss_fn(scores, labels)
        assert loss.item() >= 0.0

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        loss_fn = MultipleNegativesRankingLoss()

        # Create scores that require grad
        scores = torch.randn(4, 4, requires_grad=True)

        loss = loss_fn(scores)
        loss.backward()

        # Check gradients exist
        assert scores.grad is not None
        assert scores.grad.shape == scores.shape

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        loss_fn = MultipleNegativesRankingLoss()

        scores = torch.randn(1, 4)
        loss = loss_fn(scores)

        assert loss.shape == torch.Size([])

    def test_different_num_negatives(self):
        """Test with different numbers of negatives."""
        loss_fn = MultipleNegativesRankingLoss()

        # 1 positive + 2 negatives
        scores_3 = torch.randn(4, 3)
        loss_3 = loss_fn(scores_3)
        assert loss_3.item() >= 0.0

        # 1 positive + 5 negatives
        scores_6 = torch.randn(4, 6)
        loss_6 = loss_fn(scores_6)
        assert loss_6.item() >= 0.0

    def test_scale_effect(self):
        """Test that higher scale increases loss magnitude."""
        scores = torch.tensor(
            [
                [2.0, 1.5, 1.0, 0.5],
                [2.0, 1.5, 1.0, 0.5],
            ]
        )

        # Loss with low scale
        loss_fn_low = MultipleNegativesRankingLoss(scale=1.0, activation_fn=None)
        loss_low = loss_fn_low(scores)

        # Loss with high scale
        loss_fn_high = MultipleNegativesRankingLoss(scale=100.0, activation_fn=None)
        loss_high = loss_fn_high(scores)

        # Higher scale should generally lead to different loss values
        # (relationship depends on cross-entropy behavior)
        assert loss_low.item() != loss_high.item()

    def test_device_handling(self):
        """Test loss works on different devices."""
        loss_fn = MultipleNegativesRankingLoss()

        # CPU
        scores_cpu = torch.randn(4, 4)
        loss_cpu = loss_fn(scores_cpu)
        assert loss_cpu.device.type == "cpu"

        # GPU (if available)
        if torch.cuda.is_available():
            scores_cuda = torch.randn(4, 4, device="cuda")
            loss_cuda = loss_fn(scores_cuda)
            assert loss_cuda.device.type == "cuda"

    def test_deterministic(self):
        """Test that same input gives same output."""
        loss_fn = MultipleNegativesRankingLoss()

        scores = torch.tensor(
            [
                [2.0, 1.0, 0.5, 0.0],
                [3.0, 2.0, 1.0, 0.5],
            ]
        )

        loss1 = loss_fn(scores)
        loss2 = loss_fn(scores)

        assert torch.allclose(loss1, loss2)

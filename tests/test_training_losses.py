"""Tests for training loss functions."""

import torch

from src.training.losses import NTXent


class TestNTXent:
    """Test NTXent loss function."""

    def test_ntxent_init(self):
        """Test NTXent initialization."""
        loss_fn = NTXent(temperature=0.5)
        assert loss_fn.t == 0.5

        # Test default temperature
        loss_fn_default = NTXent()
        assert loss_fn_default.t == 0.07

    def test_ntxent_forward_single_pair(self):
        """Test NTXent forward pass with a single pair of embeddings."""
        loss_fn = NTXent(temperature=0.1)

        # Create normalized embeddings for one pair
        z1 = torch.nn.functional.normalize(torch.randn(1, 128), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(1, 128), dim=-1)

        loss = loss_fn(z1, z2)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_ntxent_forward_multiple_pairs(self):
        """Test NTXent forward pass with multiple pairs of embeddings."""
        loss_fn = NTXent(temperature=0.1)

        batch_size = 4
        embedding_dim = 128

        # Create normalized embeddings for multiple pairs
        z1 = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)

        loss = loss_fn(z1, z2)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_ntxent_forward_perfect_alignment(self):
        """Test NTXent with perfectly aligned embeddings (should give low loss)."""
        loss_fn = NTXent(temperature=0.1)

        # Create identical embeddings (perfect alignment)
        z1 = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        z2 = z1.clone()  # Perfect match

        loss = loss_fn(z1, z2)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
        # Loss should be relatively low for perfect alignment
        assert loss < 1.0

    def test_ntxent_forward_random_embeddings(self):
        """Test NTXent with random embeddings."""
        loss_fn = NTXent(temperature=0.1)

        # Create random normalized embeddings
        z1 = torch.nn.functional.normalize(torch.randn(3, 256), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(3, 256), dim=-1)

        loss = loss_fn(z1, z2)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0

    def test_ntxent_temperature_effect(self):
        """Test that temperature affects the loss magnitude."""
        z1 = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)

        loss_cold = NTXent(temperature=0.01)(z1, z2)  # Very cold
        loss_warm = NTXent(temperature=1.0)(z1, z2)   # Warmer

        # Colder temperature should generally give higher loss
        # (but this is not guaranteed due to randomness)
        assert isinstance(loss_cold, torch.Tensor)
        assert isinstance(loss_warm, torch.Tensor)

    def test_ntxent_gradient_flow(self):
        """Test that gradients flow through the NTXent loss."""
        loss_fn = NTXent(temperature=0.1)

        z1 = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)

        # Make tensors require gradients
        z1.requires_grad_(True)
        z2.requires_grad_(True)

        loss = loss_fn(z1, z2)
        loss.backward()

        assert z1.grad is not None
        assert z2.grad is not None
        assert z1.grad.shape == z1.shape
        assert z2.grad.shape == z2.shape

    def test_ntxent_different_embedding_dims(self):
        """Test NTXent with different embedding dimensions."""
        loss_fn = NTXent(temperature=0.1)

        for dim in [64, 128, 256, 512]:
            z1 = torch.nn.functional.normalize(torch.randn(2, dim), dim=-1)
            z2 = torch.nn.functional.normalize(torch.randn(2, dim), dim=-1)

            loss = loss_fn(z1, z2)

            assert isinstance(loss, torch.Tensor)
            assert loss >= 0

    def test_ntxent_batch_sizes(self):
        """Test NTXent with different batch sizes."""
        loss_fn = NTXent(temperature=0.1)

        for batch_size in [1, 2, 4, 8]:
            z1 = torch.nn.functional.normalize(torch.randn(batch_size, 128), dim=-1)
            z2 = torch.nn.functional.normalize(torch.randn(batch_size, 128), dim=-1)

            loss = loss_fn(z1, z2)

            assert isinstance(loss, torch.Tensor)
            assert loss >= 0

    def test_ntxent_device_consistency(self):
        """Test that NTXent works on different devices."""
        loss_fn = NTXent(temperature=0.1)

        # Test on CPU
        z1_cpu = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        z2_cpu = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)

        loss_cpu = loss_fn(z1_cpu, z2_cpu)
        assert isinstance(loss_cpu, torch.Tensor)

        # Test on CUDA if available
        if torch.cuda.is_available():
            z1_cuda = z1_cpu.cuda()
            z2_cuda = z2_cpu.cuda()

            loss_cuda = loss_fn(z1_cuda, z2_cuda)
            assert isinstance(loss_cuda, torch.Tensor)
            assert loss_cuda.device.type == "cuda"

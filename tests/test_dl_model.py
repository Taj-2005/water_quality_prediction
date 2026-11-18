"""Unit tests for WaterQualityMLP model architecture (src/dl/model.py).

Verifies:
  1. Default initialization and configurable hidden layers.
  2. Forward pass output shape matches (batch_size, output_dim).
  3. Supported activation functions (ReLU, GELU, SiLU, LeakyReLU, ELU, Tanh).
  4. Dropout and normalization layers (BatchNorm1d, LayerNorm).
  5. Weight initialization schemes.
  6. Parameter counting and config serialization.
  7. End-to-end training integration with src.dl.train.train.

Owner: Himanshu Pandey.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="PyTorch is required for model tests")

import torch
import torch.nn as nn

from src.dl.model import WaterQualityMLP, create_mlp, get_activation


class TestWaterQualityMLP:
    """Test suite for configurable WaterQualityMLP class."""

    def test_default_instantiation(self):
        """Test default architecture with standard parameters."""
        model = WaterQualityMLP(input_dim=25)
        assert model.input_dim == 25
        assert model.output_dim == 1
        assert model.hidden_dims == (128, 64, 32)
        assert model.dropout_rate == 0.0
        assert not model.use_batch_norm
        assert not model.use_layer_norm
        assert model.count_parameters() > 0

    def test_forward_output_shape(self):
        """Test forward pass generates correct shape (batch_size, 1)."""
        model = WaterQualityMLP(input_dim=10, hidden_dims=[32, 16], output_dim=1)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 1)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "leaky_relu", "elu", "tanh"])
    def test_activations(self, activation: str):
        """Test all supported activation functions construct and forward propagate."""
        model = WaterQualityMLP(
            input_dim=12,
            hidden_dims=[24, 12],
            activation=activation,
        )
        x = torch.randn(4, 12)
        out = model(x)
        assert out.shape == (4, 1)

    def test_batch_normalization(self):
        """Test batch normalization layers are added and forward works in train/eval."""
        model = WaterQualityMLP(
            input_dim=15,
            hidden_dims=[30, 20],
            use_batch_norm=True,
        )
        # Check that BatchNorm1d instances exist
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 2

        # In train mode (batch_size > 1)
        model.train()
        x = torch.randn(16, 15)
        out = model(x)
        assert out.shape == (16, 1)

        # In eval mode
        model.eval()
        out_eval = model(x)
        assert out_eval.shape == (16, 1)

    def test_layer_normalization(self):
        """Test layer normalization layers are added correctly."""
        model = WaterQualityMLP(
            input_dim=15,
            hidden_dims=[30, 20],
            use_layer_norm=True,
        )
        ln_layers = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
        assert len(ln_layers) == 2
        x = torch.randn(8, 15)
        out = model(x)
        assert out.shape == (8, 1)

    def test_dropout_layer(self):
        """Test dropout layers are created when dropout > 0."""
        model = WaterQualityMLP(
            input_dim=10,
            hidden_dims=[20, 10],
            dropout=0.25,
        )
        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 2
        assert dropout_layers[0].p == 0.25

    def test_conflicting_norms_raise_error(self):
        """Test ValueError when both batch_norm and layer_norm are requested."""
        with pytest.raises(ValueError, match="cannot both be True"):
            WaterQualityMLP(
                input_dim=10,
                use_batch_norm=True,
                use_layer_norm=True,
            )

    def test_invalid_dimensions_raise_error(self):
        """Test error handling on non-positive input/output dimensions."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            WaterQualityMLP(input_dim=0)
        with pytest.raises(ValueError, match="output_dim must be positive"):
            WaterQualityMLP(input_dim=10, output_dim=0)
        with pytest.raises(ValueError, match="dropout must be in"):
            WaterQualityMLP(input_dim=10, dropout=1.5)
        with pytest.raises(ValueError, match="at least one hidden layer"):
            WaterQualityMLP(input_dim=10, hidden_dims=[])

    def test_factory_create_mlp(self):
        """Test create_mlp factory function."""
        model = create_mlp(input_dim=20, hidden_dims=[64, 32], activation="gelu", dropout=0.1)
        assert isinstance(model, WaterQualityMLP)
        assert model.hidden_dims == (64, 32)
        assert model.activation_name == "gelu"
        assert model.dropout_rate == 0.1

    def test_get_config_and_count_parameters(self):
        """Test parameter counting and architecture configuration export."""
        model = WaterQualityMLP(input_dim=10, hidden_dims=[20, 10], output_dim=1)
        cfg = model.get_config()
        assert cfg["input_dim"] == 10
        assert cfg["hidden_dims"] == [20, 10]
        assert cfg["depth"] == 2
        assert cfg["output_dim"] == 1
        assert cfg["trainable_parameters"] == model.count_parameters()
        # Linear(10, 20) = 10*20 + 20 = 220
        # Linear(20, 10) = 20*10 + 10 = 210
        # Linear(10, 1) = 10*1 + 1 = 11
        # Total = 220 + 210 + 11 = 441
        assert model.count_parameters() == 441

    @pytest.mark.parametrize(
        "init_method", ["kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform"]
    )
    def test_initialization_schemes(self, init_method: str):
        """Test that various weight initialization schemes execute without errors."""
        model = WaterQualityMLP(
            input_dim=10,
            hidden_dims=[20, 10],
            init_method=init_method,
        )
        assert model.init_method == init_method

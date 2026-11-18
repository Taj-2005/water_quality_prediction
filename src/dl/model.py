"""Configurable Multi-Layer Perceptron (MLP) architecture for water quality prediction.

Provides:
  * WaterQualityMLP: PyTorch nn.Module with configurable depth, width, activations,
    dropout rates, and batch normalization.
  * get_activation: Helper resolving activation function by string name.
  * create_mlp: Convenient factory function constructing configured MLP instances.

Owner: Himanshu Pandey.
Part of Phase 2: Water Quality Prediction Deep Learning Extension.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


def get_activation(name: str | nn.Module) -> nn.Module:
    """Resolve activation function from string or module.

    Supported activation names (case-insensitive):
      - 'relu': Rectified Linear Unit (standard piecewise linear)
      - 'gelu': Gaussian Error Linear Unit (smooth non-monotonic)
      - 'silu' / 'swish': Sigmoid Linear Unit (smooth self-gated)
      - 'leaky_relu': LeakyReLU with negative slope 0.01
      - 'elu': Exponential Linear Unit
      - 'tanh': Hyperbolic Tangent
    """
    if isinstance(name, nn.Module):
        return name

    act_str = str(name).strip().lower()
    if act_str == "relu":
        return nn.ReLU()
    elif act_str == "gelu":
        return nn.GELU()
    elif act_str in ("silu", "swish"):
        return nn.SiLU()
    elif act_str in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    elif act_str == "elu":
        return nn.ELU()
    elif act_str == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(
            f"Unsupported activation '{name}'. Expected one of: "
            f"'relu', 'gelu', 'silu', 'leaky_relu', 'elu', 'tanh'."
        )


class WaterQualityMLP(nn.Module):
    """Configurable Multi-Layer Perceptron for continuous water quality index regression.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : Sequence[int] | int, default (128, 64, 32)
        Sequence of hidden layer dimensions or single int.
    activation : str | nn.Module, default 'relu'
        Activation function applied after linear layers.
    dropout : float, default 0.0
        Dropout probability applied after activations (0.0 to 1.0).
    use_batch_norm : bool, default False
        Whether to apply 1D Batch Normalization before activations.
    use_layer_norm : bool, default False
        Whether to apply 1D Layer Normalization before activations (mutually exclusive with batch_norm).
    output_dim : int, default 1
        Output dimension (defaults to 1 for continuous CCME_Values regression).
    init_method : str, default 'kaiming_normal'
        Weight initialization scheme: 'kaiming_normal', 'kaiming_uniform',
        'xavier_normal', or 'xavier_uniform'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | int = (128, 64, 32),
        activation: str | nn.Module = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        output_dim: int = 1,
        init_method: str = "kaiming_normal",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
        if use_batch_norm and use_layer_norm:
            raise ValueError("use_batch_norm and use_layer_norm cannot both be True")

        if isinstance(hidden_dims, int):
            hidden_layers = [hidden_dims]
        else:
            hidden_layers = list(hidden_dims)

        if len(hidden_layers) == 0:
            raise ValueError("hidden_dims must contain at least one hidden layer")
        for dim in hidden_layers:
            if dim <= 0:
                raise ValueError(f"All hidden dimensions must be positive, got {dim}")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_layers)
        self.activation_name = activation if isinstance(activation, str) else activation.__class__.__name__
        self.dropout_rate = float(dropout)
        self.use_batch_norm = bool(use_batch_norm)
        self.use_layer_norm = bool(use_layer_norm)
        self.output_dim = output_dim
        self.init_method = str(init_method).lower()

        # Build feedforward sequential layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_layers:
            # 1. Linear transformation
            layers.append(nn.Linear(prev_dim, h_dim))

            # 2. Normalization (Batch or Layer)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            elif self.use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))

            # 3. Non-linear activation
            layers.append(get_activation(activation))

            # 4. Dropout regularization
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            prev_dim = h_dim

        # Final regression output head (linear projection without activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize network weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply disciplined weight initialization across linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif self.init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif self.init_method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted continuous target tensor of shape (batch_size, output_dim).
        """
        return self.network(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total count of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict[str, Any]:
        """Return dictionary representation of the architecture configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "depth": len(self.hidden_dims),
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "use_layer_norm": self.use_layer_norm,
            "output_dim": self.output_dim,
            "init_method": self.init_method,
            "trainable_parameters": self.count_parameters(),
        }


def create_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] | int = (128, 64, 32),
    activation: str | nn.Module = "relu",
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    output_dim: int = 1,
    init_method: str = "kaiming_normal",
) -> WaterQualityMLP:
    """Convenience factory function returning a configured WaterQualityMLP instance."""
    return WaterQualityMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        output_dim=output_dim,
        init_method=init_method,
    )

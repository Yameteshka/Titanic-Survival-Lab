import torch
from torch import nn


class TitanicLinearClassifier(nn.Module):
    """A single-neuron logistic classifier."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

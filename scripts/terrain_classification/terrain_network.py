"""A small terrain network adapted from the PyTorch Quickstart tutorial.

Reference: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

from torch import nn


class NeuralNetwork(nn.Module):
    """Turn one cycle's sensor features into terrain class scores."""

    def __init__(self, input_size: int, class_count: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, class_count),
        )

    def forward(self, x):
        """Return raw class scores; CrossEntropyLoss handles them during training."""
        logits = self.linear_relu_stack(x)
        return logits

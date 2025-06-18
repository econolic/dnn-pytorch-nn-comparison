import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type

# Define a base class for network architectures to promote consistency
class BaseModel(nn.Module):
    def __init__(self, input_dim: int):
        super(BaseModel, self).__init__()
        assert isinstance(input_dim, int) and input_dim > 0, "Input dimension must be a positive integer"
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method")

# Specific architecture: 1-4-1
class Net1_4_1(BaseModel):
    """A simple network with one hidden layer of 4 neurons."""
    def __init__(self, input_dim: int):
        super(Net1_4_1, self).__init__(input_dim)
        self.fc1 = nn.Linear(input_dim, 4)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.input_dim, f"Input shape {x.shape[1]} does not match expected {self.input_dim}"
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x

# Specific architecture: 10-10-1
class Net10_10_1(BaseModel):
    """A network with two hidden layers of 10 neurons each."""
    def __init__(self, input_dim: int):
        super(Net10_10_1, self).__init__(input_dim)
        self.fc1 = nn.Linear(input_dim, 10)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.input_dim, f"Input shape {x.shape[1]} does not match expected {self.input_dim}"
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

# A more general, configurable network
class Net(nn.Module):
    """A simple, general-purpose two-layer network."""
    def __init__(self, D_in: int, H: int, D_out: int):
        super(Net, self).__init__()
        assert isinstance(D_in, int) and D_in > 0, "Input dimension must be a positive integer"
        assert isinstance(H, int) and H > 0, "Hidden dimension must be a positive integer"
        assert isinstance(D_out, int) and D_out > 0, "Output dimension must be a positive integer"
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor"
        assert x.shape[1] == self.linear1.in_features, \
            f"Input tensor shape {x.shape} does not match model input dimension {self.linear1.in_features}"
        x = self.linear1(x).clamp(min=0)  # ReLU activation
        x = self.linear2(x)
        return x


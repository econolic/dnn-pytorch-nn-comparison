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

# --- Test Suite ---
def test_model_initialization(model_class: Type[BaseModel], input_dim: int):
    """Tests if a model is initialized correctly."""
    model = model_class(input_dim)
    assert model.input_dim == input_dim, "Input dimension is incorrect"
    print(f"test_model_initialization for {model_class.__name__}: PASSED")

def test_forward_pass(model_class: Type[BaseModel], input_dim: int, output_dim: int):
    """Tests the forward pass of a model."""
    N = 64  # Batch size
    model = model_class(input_dim)
    x = torch.randn(N, input_dim)
    y_pred = model(x)
    assert y_pred.shape == (N, output_dim), f"Output shape {y_pred.shape} is incorrect for {model_class.__name__}"
    print(f"test_forward_pass for {model_class.__name__}: PASSED")

def test_training_loop(model_class: Type[BaseModel], input_dim: int, output_dim: int):
    """Tests a basic training loop for a model."""
    N = 64
    x = torch.randn(N, input_dim)
    y = torch.randn(N, output_dim)
    model = model_class(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    try:
        for _ in range(5):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"test_training_loop for {model_class.__name__}: PASSED")
    except Exception as e:
        print(f"test_training_loop for {model_class.__name__}: FAILED with error: {e}")

def run_all_tests():
    """Runs all tests for the PyTorch networks."""
    input_dim = 5
    print("--- Testing Net1_4_1 ---")
    test_model_initialization(Net1_4_1, input_dim)
    test_forward_pass(Net1_4_1, input_dim, 1)
    test_training_loop(Net1_4_1, input_dim, 1)

    print("\n--- Testing Net10_10_1 ---")
    test_model_initialization(Net10_10_1, input_dim)
    test_forward_pass(Net10_10_1, input_dim, 1)
    test_training_loop(Net10_10_1, input_dim, 1)
    
    print("\n--- Testing General Net ---")
    D_in, H, D_out = 10, 20, 5
    model = Net(D_in, H, D_out)
    assert model.linear1.in_features == D_in, "Input layer size is incorrect"
    assert model.linear1.out_features == H, "Hidden layer size is incorrect"
    assert model.linear2.in_features == H, "Input dimension of the second layer is incorrect"
    assert model.linear2.out_features == D_out, "Output layer size is incorrect"
    print("test_model_initialization for Net: PASSED")
    x = torch.randn(64, D_in)
    y_pred = model(x)
    assert y_pred.shape == (64, D_out), f"Output shape {y_pred.shape} is incorrect"
    print("test_forward_pass for Net: PASSED")


if __name__ == "__main__":
    run_all_tests()

    # Example usage
    input_dim = 5
    model1 = Net1_4_1(input_dim)
    model2 = Net10_10_1(input_dim)

    # Create random data for testing
    x_sample = torch.randn(10, input_dim)

    # Example forward pass
    output1 = model1(x_sample)
    output2 = model2(x_sample)

    print("\n--- Example Outputs ---")
    print("Output from Net1_4_1:", output1.shape)
    print("Output from Net10_10_1:", output2.shape)
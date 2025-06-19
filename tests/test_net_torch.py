import unittest
import torch
import torch.nn as nn
from net_torch import Net1_4_1, Net10_10_1, Net, BaseModel

class TestNetTorchModels(unittest.TestCase):
    def test_model_initialization(self) -> None:
        for model_class in (Net1_4_1, Net10_10_1):
            model = model_class(5)
            self.assertEqual(model.input_dim, 5)
            self.assertIsInstance(model, BaseModel)

    def test_forward_pass(self) -> None:
        batch = 64
        for model_class in (Net1_4_1, Net10_10_1):
            model = model_class(3)
            x = torch.randn(batch, 3)
            y = model(x)
            self.assertEqual(y.shape, (batch, 1))

    def test_training_loop(self) -> None:
        model = Net1_4_1(2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        x = torch.randn(32, 2)
        y = torch.randn(32, 1)
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(loss))

class TestGeneralNet(unittest.TestCase):
    def test_general_net(self) -> None:
        model = Net(4, 5, 2)
        x = torch.randn(10, 4)
        y = model(x)
        self.assertEqual(y.shape, (10, 2))

if __name__ == "__main__":
    unittest.main()

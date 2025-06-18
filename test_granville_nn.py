import unittest
import numpy as np
from granville_nn import f0, loss, gradient_descent, evaluate_model

class TestGranvilleNN(unittest.TestCase):
    def setUp(self) -> None:
        self.nfeatures = 2
        self.nobs = 10
        self.layers = 4
        self.params = np.random.uniform(0.1, 0.9, (self.layers, self.nfeatures))
        self.x = np.random.uniform(-1, 1, (self.nfeatures, self.nobs))
        self.y = np.random.uniform(-1, 1, self.nobs)
        self.args = {"model": "gaussian", "equalize": False}

    def test_f0_basic_functionality(self) -> None:
        output = f0(self.params, self.x, self.args)
        self.assertEqual(output.shape, (self.nobs,))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_loss_function(self) -> None:
        for mode in ("L1_abs", "L1_avg", "L2"):
            loss_val = loss(self.params, self.x, self.y, self.args, mode=mode)  # type: ignore
            self.assertIsInstance(loss_val, (float, np.floating))
            self.assertGreaterEqual(loss_val, 0)
            self.assertTrue(np.isfinite(loss_val))

    def test_gradient_descent_simple(self) -> None:
        true_params = np.random.uniform(0.2, 0.8, (self.layers, self.nfeatures))
        y_true = f0(true_params, self.x, self.args)
        params_init = np.random.uniform(0.1, 0.9, (self.layers, self.nfeatures))
        best_params, history, _ = gradient_descent(
            params_init,
            self.x,
            y_true,
            self.args,
            epochs=10,
            learning_rate=0.01,
            early_stop_patience=5,
        )
        self.assertEqual(best_params.shape, params_init.shape)
        self.assertGreater(len(history), 0)
        self.assertTrue(np.all(np.isfinite(best_params)))

    def test_model_evaluation(self) -> None:
        metrics = evaluate_model(self.params, self.x, self.y, self.args)
        required = ["mse", "mae", "rmse", "correlation", "r2"]
        for metric in required:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (float, np.floating))
            self.assertTrue(np.isfinite(metrics[metric]))
        self.assertGreaterEqual(metrics["mse"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_input_validation(self) -> None:
        with self.assertRaises(AssertionError):
            f0("invalid", self.x, self.args)  # type: ignore
        with self.assertRaises(AssertionError):
            invalid_params = np.random.uniform(0.1, 0.9, (3, self.nfeatures))
            f0(invalid_params, self.x, {"model": "gaussian"})
        with self.assertRaises(AssertionError):
            invalid_x = np.random.uniform(-1, 1, (self.nfeatures + 1, self.nobs))
            f0(self.params, invalid_x, {"model": "gaussian"})

if __name__ == "__main__":
    unittest.main()

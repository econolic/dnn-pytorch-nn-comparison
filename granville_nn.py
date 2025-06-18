"""
Implementation of a non-standard neural network based on Vincent Granville's algorithm.
Reference: https://www.datasciencecentral.com/how-to-build-and-optimize-high-performance-deep-neural-networks-from-scratch/

⚠️ WARNING: This implementation has critical performance issues!
📈 For production use, see optimized_granville_nn.py, which provides:
   - 100-1000x speedup from vectorization
   - 10x speedup from analytical gradients
   - 5-20x speedup from GPU support
   - Modern optimizers (Adam, RMSprop)
   - Batch processing and early stopping

💡 Use this version only for learning and understanding the algorithm's fundamentals.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Literal, cast

# Type alias for clarity
LossMode = Literal['L1_abs', 'L1_avg', 'L2']


def f0(params: np.ndarray, x: np.ndarray, args: dict[str, Any]) -> np.ndarray:
    """
    Computes the output of the non-standard DNN model for input data x.

    Args:
        params: Model parameters of shape (layers, nfeatures).
        x: Input data of shape (nfeatures, nobs).
        args: Dictionary with model hyperparameters.

    Returns:
        Output vector of shape (nobs,).
    """
    # region Input Validation
    assert isinstance(params, np.ndarray), f"params must be a numpy array, got {type(params)}"
    assert isinstance(x, np.ndarray), f"x must be a numpy array, got {type(x)}"
    assert isinstance(args, dict), f"args must be a dictionary, got {type(args)}"

    assert params.ndim == 2, f"params must be a 2D array, got shape {params.shape}"
    assert x.ndim == 2, f"x must be a 2D array, got shape {x.shape}"
    assert params.shape[1] == x.shape[0], (
        f"params feature count ({params.shape[1]}) must match x feature count ({x.shape[0]})"
    )

    assert np.all(np.isfinite(params)), "params contains non-finite values (NaN or Inf)"
    assert np.all(np.isfinite(x)), "x contains non-finite values (NaN or Inf)"

    model_type = args.get('model', 'gaussian')
    assert model_type in ['gaussian', 'approx. gaussian'], f"Unknown model type: {model_type}"

    nfeatures, nobs = x.shape
    layers, param_features = params.shape
    assert param_features == nfeatures, f"Parameter feature mismatch: {param_features} != {nfeatures}"
    assert layers % 4 == 0, f"layers must be divisible by 4 for Gaussian model, got {layers}"

    J = layers // 4  # Number of parameter groups (centers)
    assert J > 0, f"Number of Gaussian components must be positive, got {J}"
    # endregion

    ones = np.ones(nobs)
    z = np.zeros(nobs)

    for j in range(J):
        for k in range(nfeatures):
            theta0 = params[4 * j, k]          # Weight coefficient
            theta1 = params[4 * j + 1, k]      # Offset (bias)
            theta2 = params[4 * j + 2, k]      # Scale (skewness)            # Numerical stability checks and corrections
            assert not np.isnan(theta0), f"theta0 is NaN at position ({4*j}, {k})"
            assert not np.isnan(theta2), f"theta2 is NaN at position ({4*j+2}, {k})"
            
            # Ensure theta2 is not too close to zero by using a minimum threshold
            MIN_THETA2 = 1e-6  # Minimum allowable value for theta2
            if abs(theta2) < MIN_THETA2:
                theta2 = MIN_THETA2 if theta2 >= 0 else -MIN_THETA2

            if model_type == 'gaussian':
                # Main non-linear function: exp(-((x - theta1) / theta2)^2)
                exponent = -(((x[k, :] - theta1)) / theta2) ** 2
                # Clip exponent to prevent overflow in exp()
                exponent = np.clip(exponent, -700, 700)
                contribution = theta0 * np.exp(exponent)
            else:  # 'approx. gaussian'
                # Approximation of exponential for potential speedup
                normalized_input = (x[k, :] - theta1) / theta2
                contribution = theta0 * (1 - normalized_input ** 2)

            assert np.all(np.isfinite(contribution)), (
                f"Non-finite values in {model_type} computation at j={j}, k={k}"
            )
            z += contribution

    # Final validation of output
    assert np.all(np.isfinite(z)), "Output contains non-finite values"

    if args.get('equalize', False):
        z_min = np.min(z)
        z = z - z_min  # "Equalize" output by subtracting the minimum value

    return z


def loss(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    args: dict[str, Any],
    mode: LossMode = 'L2'
) -> float:
    """
    Computes the loss function for the current parameters on data (X, y).

    Args:
        params: Model parameters.
        X: Training input data.
        y: Training target values.
        args: Dictionary with hyperparameters.
        mode: Type of loss function ('L1_abs', 'L1_avg', 'L2').

    Returns:
        The computed loss value.
    """
    y_pred = f0(params, X, args)
    diff = y - y_pred

    if mode == 'L1_abs':
        return np.max(np.abs(diff))  # Maximum absolute error
    if mode == 'L1_avg':
        return float(np.mean(np.abs(diff)))  # Mean absolute error (MAE)
    
    # Default to 'L2'
    return float(np.dot(diff, diff) / len(diff))  # Mean squared error (MSE)


def partial_derivatives(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    args: dict[str, Any],
    loss_mode: LossMode = 'L2'
) -> np.ndarray:
    """
    Numerically estimates the gradient of the loss function for each parameter.

    Args:
        params: Model parameters.
        X: Input data.
        y: Target values.
        args: Dictionary with hyperparameters.
        loss_mode: Type of loss function.

    Returns:
        The gradient of the loss function.
    """
    eps = float(args.get('eps', 1e-6))
    layers, nfeatures = params.shape
    current_loss = loss(params, X, y, args, mode=loss_mode)
    pd = np.zeros_like(params)

    # Compute partial derivatives via eps increment
    for l in range(layers):
        for k in range(nfeatures):
            params_eps = params.copy()
            params_eps[l, k] += eps
            loss_right = loss(params_eps, X, y, args, mode=loss_mode)
            pd[l, k] = (loss_right - current_loss) / eps

    return pd


def gradient_descent(
    params_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    args: dict[str, Any],
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    loss_mode: LossMode = 'L2',
    epochs: int = 10000,
    learning_rate: float = 0.1,
    early_stop_patience: int = 100
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Runs gradient descent to optimize DNN parameters.

    Args:
        params_init: Initial model parameters.
        X: Training input data.
        y: Training target values.
        args: Dictionary with hyperparameters.
        X_val: Validation input data.
        y_val: Validation target values.
        loss_mode: Type of loss function.
        epochs: Maximum number of epochs.
        learning_rate: Learning rate.
        early_stop_patience: Patience for early stopping.

    Returns:
        A tuple containing the best parameters, training history, and validation history.
    """
    params = params_init.copy()
    best_params = params.copy()
    
    has_validation_set = X_val is not None and y_val is not None
    
    # Determine initial best loss
    X_eval = cast(np.ndarray, X_val) if has_validation_set else X
    y_eval = cast(np.ndarray, y_val) if has_validation_set else y
    best_val_loss = loss(params, X_eval, y_eval, args, mode=loss_mode)

    patience_counter = 0
    history: list[float] = []
    val_history: list[float] = []

    for epoch in range(epochs):
        val_loss: float | None = None
        # Perform a gradient descent step
        pd = partial_derivatives(params, X, y, args, loss_mode=loss_mode)
        params -= learning_rate * pd        # Clip parameters to a safe range to avoid numerical issues
        # Use small epsilon to avoid exactly zero values which cause division by zero
        MIN_PARAM = 1e-6
        MAX_PARAM = 1.0 - 1e-6
        params = np.clip(params, MIN_PARAM, MAX_PARAM)

        # Calculate and record training loss
        train_loss = loss(params, X, y, args, mode=loss_mode)
        history.append(train_loss)

        if has_validation_set:
            val_loss = loss(params, cast(np.ndarray, X_val), cast(np.ndarray, y_val), args, mode=loss_mode)
            val_history.append(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            # If no validation set, best parameters are the latest
            best_params = params.copy()

        if (epoch + 1) % 1000 == 0:
            val_msg = f", val_loss: {val_loss:.6f}" if val_loss is not None else ""
            print(f"Epoch {epoch + 1:5d}: loss: {train_loss:.6f}{val_msg}")

    return best_params, history, val_history


def predict(params: np.ndarray, X: np.ndarray, args: dict[str, Any]) -> np.ndarray:
    """
    Makes predictions using the trained model.

    Args:
        params: Trained model parameters.
        X: Input data for prediction.
        args: Dictionary with hyperparameters.

    Returns:
        Predicted values.
    """
    return f0(params, X, args)


def evaluate_model(
    params: np.ndarray, X: np.ndarray, y: np.ndarray, args: dict[str, Any]
) -> dict[str, float]:
    """
    Evaluates the model quality on test data.

    Args:
        params: Trained model parameters.
        X: Test input data.
        y: Test target values.
        args: Dictionary with hyperparameters.

    Returns:
        A dictionary with quality metrics.
    """
    y_pred = predict(params, X, args)

    mse = float(np.mean((y - y_pred) ** 2))
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(mse))

    # Correlation coefficient with validity check
    corr_matrix = np.corrcoef(y, y_pred)
    if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
        correlation = float(corr_matrix[0, 1])
    else:
        correlation = 0.0

    # R² coefficient of determination
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "r2": r2,
    }


# --- Test Suite ---
import unittest


class TestGranvilleNN(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.nfeatures = 2
        self.nobs = 10
        self.layers = 4
        self.params = np.random.uniform(0.1, 0.9, (self.layers, self.nfeatures))
        self.x = np.random.uniform(-1, 1, (self.nfeatures, self.nobs))
        self.y = np.random.uniform(-1, 1, self.nobs)
        self.args = {"model": "gaussian", "equalize": False}

    def test_f0_basic_functionality(self):
        """Test basic functionality of f0 function."""
        output = f0(self.params, self.x, self.args)
        self.assertEqual(output.shape, (self.nobs,))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_loss_function(self):
        """Test loss function with different modes."""
        for mode in ("L1_abs", "L1_avg", "L2"):
            loss_val = loss(self.params, self.x, self.y, self.args, mode=mode) # type: ignore
            self.assertIsInstance(loss_val, (float, np.floating))
            self.assertGreaterEqual(loss_val, 0)
            self.assertTrue(np.isfinite(loss_val))

    def test_gradient_descent_simple(self):
        """Test gradient descent with simple case."""
        true_params = np.random.uniform(0.2, 0.8, (self.layers, self.nfeatures))
        y_true = f0(true_params, self.x, self.args)

        params_init = np.random.uniform(0.1, 0.9, (self.layers, self.nfeatures))

        best_params, history, val_history = gradient_descent(
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

    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        metrics = evaluate_model(self.params, self.x, self.y, self.args)

        required_metrics = ["mse", "mae", "rmse", "correlation", "r2"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (float, np.floating))
            self.assertTrue(np.isfinite(metrics[metric]))

        self.assertGreaterEqual(metrics["mse"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_input_validation(self):
        """Test input validation and error handling."""
        with self.assertRaisesRegex(AssertionError, "params must be a numpy array"):
            f0("invalid", self.x, self.args)  # type: ignore

        with self.assertRaisesRegex(AssertionError, "layers must be divisible by 4"):
            invalid_params = np.random.uniform(0.1, 0.9, (3, self.nfeatures))
            f0(invalid_params, self.x, {"model": "gaussian"})

        with self.assertRaisesRegex(AssertionError, "must match x feature count"):
            invalid_x = np.random.uniform(-1, 1, (self.nfeatures + 1, self.nobs))
            f0(self.params, invalid_x, {"model": "gaussian"})


if __name__ == "__main__":
    unittest.main()



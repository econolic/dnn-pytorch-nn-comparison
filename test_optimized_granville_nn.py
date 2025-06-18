import unittest
import numpy as np
from optimized_granville_nn import OptimizedGranvilleDNN, OptimizationConfig, TrainingConfig

class TestOptimizedGranvilleDNN(unittest.TestCase):
    def setUp(self):
        """Set up a basic dataset and model for testing."""
        self.X_train = np.random.rand(100, 3).astype(np.float32)
        self.y_train = np.random.rand(100).astype(np.float32)
        self.X_test = np.random.rand(20, 3).astype(np.float32)
        self.y_test = np.random.rand(20).astype(np.float32)

        self.model = OptimizedGranvilleDNN(centers=5, random_state=42)

    def test_model_initialization(self):
        """Test that the model and its components are initialized correctly."""
        self.assertEqual(self.model.centers, 5)
        self.assertIsNone(self.model.parameters)
        self.assertIsNone(self.model.n_features)
        self.assertIsInstance(self.model.opt_config, OptimizationConfig)
        self.assertIsInstance(self.model.train_config, TrainingConfig)

        # Test initialization with custom configs
        opt_config = OptimizationConfig(optimizer='sgd', learning_rate=0.1)
        train_config = TrainingConfig(epochs=50, batch_size=16)
        model = OptimizedGranvilleDNN(centers=10, random_state=42, optimization_config=opt_config, training_config=train_config)
        self.assertEqual(model.centers, 10)
        self.assertEqual(model.opt_config.optimizer, 'sgd')
        self.assertEqual(model.opt_config.learning_rate, 0.1)
        self.assertEqual(model.train_config.epochs, 50)
        self.assertEqual(model.train_config.batch_size, 16)

    def test_parameter_initialization(self):
        """Test the internal parameter initialization."""
        self.model._initialize_parameters(n_features=3)
        self.assertEqual(self.model.n_features, 3)
        self.assertIsNotNone(self.model.parameters)
        assert self.model.parameters is not None, "Parameters should be initialized"
        self.assertEqual(self.model.parameters.shape, (4 * 5, 3))
        # Check that scale parameters are positive
        for j in range(self.model.centers):
            scale_idx = 4 * j + 2
            self.assertTrue(np.all(self.model.parameters[scale_idx] > 0))

    def test_fit_and_predict(self):
        """Test the full training and prediction cycle."""
        self.model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.parameters)
        self.assertGreater(len(self.model.history['loss']), 0)

        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape, (20,))
        self.assertEqual(predictions.dtype, np.float32)

    def test_score(self):
        """Test the scoring functionality."""
        self.model.fit(self.X_train, self.y_train)
        
        # Test R2 score
        r2_score = self.model.score(self.X_test, self.y_test, metric='r2')
        self.assertIsInstance(r2_score, float)
        self.assertLessEqual(r2_score, 1.0)

        # Test MSE score
        mse_score = self.model.score(self.X_test, self.y_test, metric='mse')
        self.assertIsInstance(mse_score, float)
        self.assertGreaterEqual(mse_score, 0.0)

        # Test MAE score
        mae_score = self.model.score(self.X_test, self.y_test, metric='mae')
        self.assertIsInstance(mae_score, float)
        self.assertGreaterEqual(mae_score, 0.0)

    def test_input_validation(self):
        """Test that the model raises errors for invalid input."""
        # Test fit() validation
        with self.assertRaises(AssertionError):
            self.model.fit(self.X_train.tolist(), self.y_train) # Invalid X type
        with self.assertRaises(AssertionError):
            self.model.fit(self.X_train, self.y_train.reshape(-1, 1)) # Invalid y shape
        with self.assertRaises(AssertionError):
            self.model.fit(self.X_train[:10], self.y_train) # Mismatched lengths

        self.model.fit(self.X_train, self.y_train)

        # Test predict() validation
        with self.assertRaises(AssertionError):
            self.model.predict(self.X_test.tolist()) # Invalid X type
        with self.assertRaises(AssertionError):
            self.model.predict(self.X_test[:, :2]) # Incorrect feature count

        # Test score() validation
        with self.assertRaises(AssertionError):
            self.model.score(self.X_test, self.y_test.reshape(-1, 1)) # Invalid y shape
        with self.assertRaises(AssertionError):
            self.model.score(self.X_test, self.y_test, metric='invalid_metric') # Invalid metric

    def test_early_stopping(self):
        """Test the early stopping functionality."""
        train_config = TrainingConfig(epochs=2000, validation_split=0.2, early_stopping_patience=10, verbose=0)
        model = OptimizedGranvilleDNN(centers=5, random_state=42, training_config=train_config)
        model.fit(self.X_train, self.y_train)

        # Check that training stopped before the max epochs
        self.assertLess(len(model.history['loss']), 2000)
        self.assertGreater(len(model.history['loss']), 10)

if __name__ == '__main__':
    unittest.main()

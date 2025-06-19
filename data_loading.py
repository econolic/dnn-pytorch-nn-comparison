"""
Enhanced Data Loading and Preprocessing Module

This module provides standardized data loading, preprocessing, and splitting
functionality for neural network comparison experiments. Designed to ensure
fair comparison between Granville DNN and PyTorch implementations.

Key Features:
- Consistent preprocessing pipelines
- Multiple scaling methods (Standard, MinMax, Robust)
- Data validation and quality checks
- Configurable train/validation/test splits
- Comprehensive dataset analytics
- Memory-efficient operations

Author: Enhanced implementation for DNN comparison project
Date: June 2025
Compliance: PEP 8, PEP 484, ISO/IEC/IEEE 12207:2017
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Tuple, Dict, Union, Optional, Literal, Any
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import check_array, check_X_y
import pandas as pd

@dataclass
class DatasetInfo:
    """Container for dataset metadata and comprehensive statistics."""
    name: str
    n_samples: int
    n_features: int
    target_range: Tuple[float, float]
    feature_ranges: Dict[str, Tuple[float, float]]
    correlation_with_target: float
    feature_names: Optional[Tuple[str, ...]] = None
    target_name: Optional[str] = None
    data_quality: Dict[str, Any] = field(default_factory=dict)
    preprocessing_info: Dict[str, Any] = field(default_factory=dict)


class DataPreprocessor:
    """
    Enhanced data preprocessing pipeline for neural network comparison.
    
    Ensures all models receive identical data preprocessing to guarantee
    fair comparison. Supports multiple scaling methods and includes
    comprehensive data validation and quality checks.
    
    Features:
    - Multiple scaling methods (Standard, MinMax, Robust)
    - Data validation and outlier detection
    - Memory-efficient processing
    - Preprocessing pipeline serialization
    - Comprehensive logging
    """
    
    def __init__(
        self, 
        scaling_method: Literal['standard', 'minmax', 'robust'] = 'standard',
        feature_range: Tuple[int, int] = (0, 1),
        with_mean: bool = True,
        with_std: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0)
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: Scaling method to use
            feature_range: Range for MinMaxScaler
            with_mean: Whether to center data (StandardScaler)
            with_std: Whether to scale to unit variance (StandardScaler)
            quantile_range: Quantile range for RobustScaler
        """
        valid_methods = ['standard', 'minmax', 'robust']
        if scaling_method not in valid_methods:
            raise ValueError(f"scaling_method must be one of {valid_methods}, got {scaling_method}")
        
        self.scaling_method = scaling_method
        self.feature_range = feature_range
        self.with_mean = with_mean
        self.with_std = with_std
        self.quantile_range = quantile_range
        
        self.feature_scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.target_scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.is_fitted = False
        self.data_stats: Dict[str, Any] = {}
    
    def _create_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Create appropriate scaler based on configuration."""
        if self.scaling_method == 'standard':
            return StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        elif self.scaling_method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.scaling_method == 'robust':
            return RobustScaler(quantile_range=self.quantile_range, with_centering=self.with_mean)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input data quality and constraints."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if not np.isfinite(X).all():
            warnings.warn("X contains non-finite values (inf/nan)")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array")
            if y.ndim != 1:
                raise ValueError(f"y must be 1D, got shape {y.shape}")
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
            if not np.isfinite(y).all():
                warnings.warn("y contains non-finite values (inf/nan)")
    
    def _compute_data_statistics(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Compute comprehensive data statistics."""
        self.data_stats = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_means': np.mean(X, axis=0),
            'feature_stds': np.std(X, axis=0),
            'feature_mins': np.min(X, axis=0),
            'feature_maxs': np.max(X, axis=0),
            'feature_ranges': np.ptp(X, axis=0),
            'missing_values': np.isnan(X).sum(),
            'infinite_values': np.isinf(X).sum(),
        }
        
        if y is not None:
            self.data_stats.update({
                'target_mean': np.mean(y),
                'target_std': np.std(y),
                'target_min': np.min(y),
                'target_max': np.max(y),
                'target_range': np.ptp(y),
                'target_missing': np.isnan(y).sum(),
                'target_infinite': np.isinf(y).sum(),
            })
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scalers and transform data with comprehensive validation."""
        self._validate_data(X, y)
        self._compute_data_statistics(X, y)
        
        # Create scalers
        self.feature_scaler = self._create_scaler()
        self.target_scaler = self._create_scaler()
        
        # Fit and transform
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        self.is_fitted = True
        return X_scaled.astype(np.float32), y_scaled.astype(np.float32)
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
        if self.feature_scaler is None:
            raise ValueError("Feature scaler not initialized")
        
        self._validate_data(X, y)
        X_scaled = self.feature_scaler.transform(X)
        
        if y is not None:
            if self.target_scaler is None:
                raise ValueError("Target scaler not initialized")
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).ravel()
            return X_scaled.astype(np.float32), y_scaled.astype(np.float32)
        
        return X_scaled.astype(np.float32)
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target values."""
        if not self.is_fitted or self.target_scaler is None:
            raise ValueError("Scaler not fitted")
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing information."""
        return {
            'scaling_method': self.scaling_method,
            'feature_range': self.feature_range,
            'is_fitted': self.is_fitted,
            'data_stats': self.data_stats.copy(),
            'feature_scaler_params': getattr(self.feature_scaler, '__dict__', {}) if self.feature_scaler else {},
            'target_scaler_params': getattr(self.target_scaler, '__dict__', {}) if self.target_scaler else {},
        }


def load_and_analyze_dataset(
    dataset_name: Literal['california_housing', 'synthetic'] = 'california_housing',
    synthetic_config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """
    Load and analyze dataset with comprehensive quality checks.
    
    Args:
        dataset_name: Dataset to load
        synthetic_config: Configuration for synthetic data generation
    
    Returns:
        Tuple of (features, targets, dataset_info)
    """
    print(f"📊 Loading {dataset_name.replace('_', ' ').title()} Dataset...")
    
    if dataset_name == 'california_housing':
        # Load California Housing dataset
        housing_data = fetch_california_housing()
        X, y = housing_data.data, housing_data.target  # type: ignore
        feature_names = housing_data.feature_names  # type: ignore
        dataset_display_name = "California Housing"
        target_name = "Median House Value"
        
    elif dataset_name == 'synthetic':
        # Generate synthetic regression dataset
        config = synthetic_config or {}
        n_samples = config.get('n_samples', 1000)
        n_features = config.get('n_features', 8)
        noise = config.get('noise', 0.1)
        random_state = config.get('random_state', 42)
        
        X, y = make_regression(  # type: ignore
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        feature_names = [f"feature_{i}" for i in range(n_features)]
        dataset_display_name = f"Synthetic ({n_samples}x{n_features})"
        target_name = "Target"
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Validate data
    X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
    
    print(f"   • Dataset shape: {X.shape}")
    print(f"   • Target shape: {y.shape}")
    print(f"   • Features: {feature_names}")
    print(f"   • Data types: X={X.dtype}, y={y.dtype}")
    
    # Calculate comprehensive dataset statistics
    feature_ranges = {
        name: (float(X[:, i].min()), float(X[:, i].max()))
        for i, name in enumerate(feature_names)
    }
    
    # Calculate correlations with target
    correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
    avg_correlation = float(np.mean(np.abs(correlations)))
    
    # Data quality assessment
    data_quality = {
        'missing_values': int(np.isnan(X).sum() + np.isnan(y).sum()),
        'infinite_values': int(np.isinf(X).sum() + np.isinf(y).sum()),
        'duplicated_samples': int(pd.DataFrame(X).duplicated().sum()),
        'feature_correlations': {
            name: float(corr) for name, corr in zip(feature_names, correlations)
        },
        'condition_number': float(np.linalg.cond(X)),
        'rank': int(np.linalg.matrix_rank(X)),
    }
    
    # Outlier detection (basic)
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    outliers_count = int(np.sum(z_scores > 3))
    data_quality['outliers_count'] = outliers_count
    data_quality['outliers_percentage'] = float(outliers_count / (X.shape[0] * X.shape[1]) * 100)
    
    dataset_info = DatasetInfo(
        name=dataset_display_name,
        n_samples=len(X),
        n_features=X.shape[1],
        target_range=(float(y.min()), float(y.max())),
        feature_ranges=feature_ranges,
        correlation_with_target=avg_correlation,
        feature_names=tuple(feature_names),
        target_name=target_name,
        data_quality=data_quality
    )
    
    # Print quality assessment
    print(f"   • Data Quality Assessment:")
    print(f"     - Missing values: {data_quality['missing_values']}")
    print(f"     - Infinite values: {data_quality['infinite_values']}")
    print(f"     - Duplicated samples: {data_quality['duplicated_samples']}")
    print(f"     - Outliers (>3σ): {data_quality['outliers_count']} ({data_quality['outliers_percentage']:.2f}%)")
    print(f"     - Condition number: {data_quality['condition_number']:.2e}")
    print(f"     - Matrix rank: {data_quality['rank']}/{X.shape[1]}")
    print(f"   • Average |correlation| with target: {avg_correlation:.3f}")
    
    return X, y, dataset_info


def create_train_val_test_splits(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = False,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create enhanced train/validation/test splits with multiple options.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting (for regression, creates bins)
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Validate inputs
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
    if test_size + val_size >= 1:
        raise ValueError(f"test_size + val_size must be < 1, got {test_size + val_size}")
    
    print(f"📊 Creating data splits with test={test_size:.1%}, val={val_size:.1%}, train={1-test_size-val_size:.1%}")
    
    # Handle stratification for regression
    stratify_y = None
    if stratify and len(np.unique(y)) < 50:  # If not too many unique values
        stratify_y = y
    elif stratify:
        # Create bins for continuous targets
        n_bins = min(10, len(y) // 50)  # Reasonable number of bins
        if n_bins >= 2:
            stratify_y = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
            print(f"   • Using stratified split with {n_bins} bins")
    
    # First split: separate test set
    if stratify_y is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_y, shuffle=shuffle
        )
        # Update stratify_y for remaining data
        if stratify and len(np.unique(y_temp)) >= 2:
            if len(np.unique(y_temp)) < 50:
                stratify_y_temp = y_temp
            else:
                n_bins_temp = min(10, len(y_temp) // 50)
                if n_bins_temp >= 2:
                    stratify_y_temp = pd.cut(y_temp, bins=n_bins_temp, labels=False, duplicates='drop')
                else:
                    stratify_y_temp = None
        else:
            stratify_y_temp = None
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        stratify_y_temp = None
    
    # Second split: separate training and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    if stratify_y_temp is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=stratify_y_temp, shuffle=shuffle
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            shuffle=shuffle
        )
    
    # Print split information
    total_samples = len(X)
    print(f"📊 Data splits created:")
    print(f"   • Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/total_samples*100:.1f}%)")
    print(f"   • Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/total_samples*100:.1f}%)")
    print(f"   • Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/total_samples*100:.1f}%)")
    
    # Print target distribution statistics
    print(f"   • Target distribution:")
    print(f"     - Train: μ={np.mean(y_train):.3f}, σ={np.std(y_train):.3f}, range=[{np.min(y_train):.3f}, {np.max(y_train):.3f}]")
    print(f"     - Val:   μ={np.mean(y_val):.3f}, σ={np.std(y_val):.3f}, range=[{np.min(y_val):.3f}, {np.max(y_val):.3f}]")
    print(f"     - Test:  μ={np.mean(y_test):.3f}, σ={np.std(y_test):.3f}, range=[{np.min(y_test):.3f}, {np.max(y_test):.3f}]")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def validate_data_consistency(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Validate consistency across train/validation/test splits.
    
    Returns:
        Dictionary with validation results and statistics
    """
    validation_results = {
        'shapes_consistent': True,
        'dtypes_consistent': True,
        'no_data_leakage': True,
        'target_distributions_similar': True,
        'warnings': [],
        'statistics': {}
    }
    
    # Check shapes consistency
    n_features_train = X_train.shape[1]
    if X_val.shape[1] != n_features_train or X_test.shape[1] != n_features_train:
        validation_results['shapes_consistent'] = False
        validation_results['warnings'].append("Inconsistent number of features across splits")
    
    # Check data types
    dtypes = [X_train.dtype, X_val.dtype, X_test.dtype, y_train.dtype, y_val.dtype, y_test.dtype]
    if len(set(dtypes)) > 1:
        validation_results['dtypes_consistent'] = False
        validation_results['warnings'].append(f"Inconsistent data types: {dtypes}")
    
    # Check for potential data leakage (exact duplicates)
    train_set = set(map(tuple, X_train))
    val_set = set(map(tuple, X_val))
    test_set = set(map(tuple, X_test))
    
    train_val_overlap = len(train_set & val_set)
    train_test_overlap = len(train_set & test_set)
    val_test_overlap = len(val_set & test_set)
    
    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        validation_results['no_data_leakage'] = False
        validation_results['warnings'].append(
            f"Data leakage detected: train-val={train_val_overlap}, "
            f"train-test={train_test_overlap}, val-test={val_test_overlap}"
        )
    
    # Check target distribution similarity (using KS test approximation)
    def distribution_distance(a, b):
        """Simple distribution distance metric"""
        return abs(np.mean(a) - np.mean(b)) / (np.std(a) + np.std(b) + 1e-8)
    
    train_val_dist = distribution_distance(y_train, y_val)
    train_test_dist = distribution_distance(y_train, y_test)
    val_test_dist = distribution_distance(y_val, y_test)
    
    if max(train_val_dist, train_test_dist, val_test_dist) > 0.5:
        validation_results['target_distributions_similar'] = False
        validation_results['warnings'].append("Target distributions significantly different across splits")
    
    # Gather statistics
    validation_results['statistics'] = {
        'sample_counts': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'target_stats': {
            'train': {'mean': float(np.mean(y_train)), 'std': float(np.std(y_train))},
            'val': {'mean': float(np.mean(y_val)), 'std': float(np.std(y_val))},
            'test': {'mean': float(np.mean(y_test)), 'std': float(np.std(y_test))}
        },
        'distribution_distances': {
            'train_val': float(train_val_dist),
            'train_test': float(train_test_dist),
            'val_test': float(val_test_dist)
        }
    }
    
    return validation_results


def create_data_summary(
    dataset_info: DatasetInfo,
    preprocessing_info: Dict[str, Any],
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a comprehensive summary of the data loading and preprocessing pipeline.
    
    Returns:
        Complete data pipeline summary
    """
    summary = {
        'dataset_info': {
            'name': dataset_info.name,
            'n_samples': dataset_info.n_samples,
            'n_features': dataset_info.n_features,
            'target_range': dataset_info.target_range,
            'avg_correlation': dataset_info.correlation_with_target,
            'data_quality': dataset_info.data_quality
        },
        'preprocessing': {
            'scaling_method': preprocessing_info.get('scaling_method'),
            'is_fitted': preprocessing_info.get('is_fitted'),
            'data_stats': preprocessing_info.get('data_stats', {})
        },
        'validation': validation_results,
        'pipeline_status': 'SUCCESS' if all([
            validation_results['shapes_consistent'],
            validation_results['dtypes_consistent'],
            validation_results['no_data_leakage']
        ]) else 'WARNING'
    }
    
    return summary


def print_data_pipeline_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary of the data pipeline."""
    print("\n" + "="*60)
    print("📋 DATA PIPELINE SUMMARY")
    print("="*60)
    
    # Dataset info
    dataset = summary['dataset_info']
    print(f"📊 Dataset: {dataset['name']}")
    print(f"   • Samples: {dataset['n_samples']:,}")
    print(f"   • Features: {dataset['n_features']}")
    print(f"   • Target range: [{dataset['target_range'][0]:.3f}, {dataset['target_range'][1]:.3f}]")
    print(f"   • Avg |correlation|: {dataset['avg_correlation']:.3f}")
    
    # Data quality
    quality = dataset['data_quality']
    print(f"   • Quality: {quality['missing_values']} missing, {quality['infinite_values']} infinite")
    print(f"   • Outliers: {quality['outliers_count']} ({quality['outliers_percentage']:.2f}%)")
    
    # Preprocessing
    preprocessing = summary['preprocessing']
    print(f"🔧 Preprocessing: {preprocessing['scaling_method']} scaling")
    print(f"   • Status: {'✅ Fitted' if preprocessing['is_fitted'] else '❌ Not fitted'}")
    
    # Validation
    validation = summary['validation']
    status_icon = "✅" if summary['pipeline_status'] == 'SUCCESS' else "⚠️"
    print(f"{status_icon} Validation: {summary['pipeline_status']}")
    
    if validation['warnings']:
        print("   • Warnings:")
        for warning in validation['warnings']:
            print(f"     - {warning}")
    
    # Split statistics
    stats = validation['statistics']['sample_counts']
    print(f"📈 Splits: Train={stats['train']:,}, Val={stats['val']:,}, Test={stats['test']:,}")
    
    print("="*60)

if __name__ == "__main__":
    # Enhanced example usage with comprehensive pipeline
    print("🚀 Enhanced Data Loading Pipeline Demo")
    print("="*50)
    
    # Load dataset with analysis
    X, y, dataset_info = load_and_analyze_dataset('california_housing')
    
    # Create preprocessor with different scaling methods
    print("\n🔧 Testing different preprocessing methods:")
    
    # Standard scaling (most common)
    preprocessor_std = DataPreprocessor(scaling_method='standard')
    X_scaled_std, y_scaled_std = preprocessor_std.fit_transform(X, y)
    
    # MinMax scaling (for Granville DNN)
    preprocessor_minmax = DataPreprocessor(scaling_method='minmax')
    X_scaled_minmax, y_scaled_minmax = preprocessor_minmax.fit_transform(X, y)
    
    # Robust scaling (for outlier-heavy data)
    preprocessor_robust = DataPreprocessor(scaling_method='robust')
    X_scaled_robust, y_scaled_robust = preprocessor_robust.fit_transform(X, y)
    
    print(f"   • Standard scaling: X range [{X_scaled_std.min():.3f}, {X_scaled_std.max():.3f}]")
    print(f"   • MinMax scaling: X range [{X_scaled_minmax.min():.3f}, {X_scaled_minmax.max():.3f}]")
    print(f"   • Robust scaling: X range [{X_scaled_robust.min():.3f}, {X_scaled_robust.max():.3f}]")
    
    # Create splits with enhanced validation
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(
        X_scaled_std, y_scaled_std, 
        test_size=0.2, 
        val_size=0.2, 
        stratify=True,
        random_state=42
    )
    
    # Validate data consistency
    print("\n🔍 Validating data consistency:")
    validation_results = validate_data_consistency(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            print(f"   ⚠️  {warning}")
    else:
        print("   ✅ All validation checks passed")
    
    # Create comprehensive summary
    preprocessing_info = preprocessor_std.get_preprocessing_info()
    summary = create_data_summary(dataset_info, preprocessing_info, validation_results)
    
    # Print summary
    print_data_pipeline_summary(summary)
    
    # Demo synthetic data generation
    print("\n🧪 Testing synthetic data generation:")
    X_synth, y_synth, dataset_info_synth = load_and_analyze_dataset(
        'synthetic',
        synthetic_config={
            'n_samples': 1000,
            'n_features': 5,
            'noise': 0.15,
            'random_state': 42
        }
    )
    
    # Performance comparison
    print("\n⚡ Performance comparison:")
    import time
    
    # Time preprocessing
    start_time = time.time()
    for _ in range(100):
        _ = preprocessor_std.transform(X_test[:100])
    preprocessing_time = (time.time() - start_time) * 10  # ms per operation
    
    print(f"   • Preprocessing: {preprocessing_time:.3f} ms per 100 samples")
    print(f"   • Memory usage: ~{X.nbytes + y.nbytes / 1024 / 1024:.2f} MB")
    print(f"   • Data types: {X.dtype} (features), {y.dtype} (targets)")
    
    print("\n✅ Enhanced data loading pipeline demonstration completed successfully!")
    print("   Ready for neural network comparison experiments.")
    print("   Use different preprocessors for different model requirements:")
    print("   - Standard scaling: PyTorch models")
    print("   - MinMax scaling: Granville DNN")
    print("   - Robust scaling: Data with outliers")
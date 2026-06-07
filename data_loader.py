"""
Data loading, preprocessing, and feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils import get_logger, load_config, set_seed

logger = get_logger("data_loader")

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class DataPipeline:
    """End-to-end data loading and preprocessing pipeline."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.data_cfg = self.config.get("data", {})
        self.prep_cfg = self.config.get("preprocessing", {})

        scaler_name = self.prep_cfg.get("scaler", "standard")
        self.scaler = SCALERS.get(scaler_name, StandardScaler)()
        self.feature_names = None

    def load(self) -> pd.DataFrame:
        """Load dataset from configured source."""
        source = self.data_cfg.get("source", "synthetic")

        if source == "synthetic":
            return self._generate_synthetic()
        elif Path(source).exists():
            logger.info(f"Loading dataset from {source}")
            return pd.read_csv(source)
        else:
            logger.warning(f"Source '{source}' not found. Falling back to synthetic data.")
            return self._generate_synthetic()

    def _generate_synthetic(self) -> pd.DataFrame:
        """Generate a realistic synthetic dataset for anomaly detection."""
        seed = self.config.get("experiment", {}).get("seed", 42)
        set_seed(seed)

        n_samples = self.data_cfg.get("n_samples", 2000)
        n_features = self.data_cfg.get("n_features", 30)
        noise = self.data_cfg.get("noise_ratio", 0.05)

        logger.info(f"Generating synthetic dataset: {n_samples} samples, {n_features} features")

        # Normal traffic features (class 0)
        n_normal = int(n_samples * 0.7)
        n_anomaly = n_samples - n_normal

        X_normal = np.random.randn(n_normal, n_features) * 0.5
        X_anomaly = np.random.randn(n_anomaly, n_features) * 1.5 + np.random.choice([-2, 2], size=(n_anomaly, n_features))

        X = np.vstack([X_normal, X_anomaly])
        y = np.array([0] * n_normal + [1] * n_anomaly)

        # Add interaction features
        X[:, -1] = X[:, 0] * X[:, 1] + np.random.randn(n_samples) * 0.1
        X[:, -2] = np.abs(X[:, 2]) + X[:, 3] ** 2

        # Add label noise
        noise_idx = np.random.choice(n_samples, size=int(n_samples * noise), replace=False)
        y[noise_idx] = 1 - y[noise_idx]

        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        X, y = X[shuffle_idx], y[shuffle_idx]

        self.feature_names = [f"feat_{i:02d}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=self.feature_names)
        df["label"] = y

        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full preprocessing: handle missing values, scale, and split."""
        logger.info("Running preprocessing pipeline...")

        # Handle missing values
        strategy = self.prep_cfg.get("handle_missing", "median")
        if strategy == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        else:
            df = df.dropna()

        # Split features and labels
        X = df.drop("label", axis=1).values
        y = df["label"].values
        self.feature_names = list(df.drop("label", axis=1).columns)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        test_size = self.data_cfg.get("test_size", 0.2)
        stratify = y if self.data_cfg.get("stratify", True) else None
        seed = self.config.get("experiment", {}).get("seed", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=seed, stratify=stratify
        )

        logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full data pipeline."""
        df = self.load()
        return self.preprocess(df)


if __name__ == "__main__":
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.run()
    print(f"\nPipeline complete: Train={X_train.shape}, Test={X_test.shape}")

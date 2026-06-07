"""
Model definitions, training, and inference.

Supports multiple classifiers with a unified interface for
fair comparison and benchmarking.
"""

import numpy as np
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from utils import get_logger, load_config, timer

logger = get_logger("model")


class ModelFactory:
    """Factory for creating configured model instances."""

    @staticmethod
    def create(model_type: str, config: Optional[dict] = None) -> Any:
        """Create a model instance from config."""
        cfg = (config or {}).get("models", {}).get(model_type, {})

        builders = {
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=cfg.get("n_estimators", 200),
                max_depth=cfg.get("max_depth", 15),
                min_samples_split=cfg.get("min_samples_split", 5),
                n_jobs=cfg.get("n_jobs", -1),
                random_state=config.get("experiment", {}).get("seed", 42),
            ),
            "gradient_boosting": lambda: GradientBoostingClassifier(
                n_estimators=cfg.get("n_estimators", 200),
                max_depth=cfg.get("max_depth", 6),
                learning_rate=cfg.get("learning_rate", 0.1),
                subsample=cfg.get("subsample", 0.8),
                random_state=config.get("experiment", {}).get("seed", 42),
            ),
            "svm": lambda: SVC(
                kernel=cfg.get("kernel", "rbf"),
                C=cfg.get("C", 1.0),
                gamma=cfg.get("gamma", "scale"),
                probability=True,
                random_state=config.get("experiment", {}).get("seed", 42),
            ),
            "mlp": lambda: MLPClassifier(
                hidden_layer_sizes=tuple(cfg.get("hidden_layers", [256, 128, 64])),
                activation=cfg.get("activation", "relu"),
                max_iter=cfg.get("max_iter", 500),
                early_stopping=cfg.get("early_stopping", True),
                validation_fraction=cfg.get("validation_fraction", 0.1),
                random_state=config.get("experiment", {}).get("seed", 42),
            ),
        }

        if model_type not in builders:
            raise ValueError(f"Unknown model: {model_type}. Available: {list(builders.keys())}")

        return builders[model_type]()


class Trainer:
    """Handles model training and cross-validation."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.eval_cfg = self.config.get("evaluation", {})

    @timer
    def train(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """Train a model and return it."""
        logger.info(f"Training {type(model).__name__} on {X_train.shape[0]} samples...")
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_acc:.4f}")
        return model

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Run cross-validation if enabled in config."""
        cv_cfg = self.eval_cfg.get("cross_validation", {})
        if not cv_cfg.get("enabled", False):
            return {}

        folds = cv_cfg.get("folds", 5)
        logger.info(f"Running {folds}-fold cross-validation...")

        scores = cross_val_score(model, X, y, cv=folds, scoring="f1_weighted", n_jobs=-1)
        result = {
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "cv_scores": scores.tolist(),
        }
        logger.info(f"CV F1: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        return result

    def train_all(self, X_train, y_train) -> Dict[str, Any]:
        """Train all configured models."""
        models = {}
        model_types = list(self.config.get("models", {}).keys())

        for model_type in model_types:
            logger.info(f"\n========================================")
            logger.info(f"  {model_type.upper()}")
            logger.info(f"========================================")

            model = ModelFactory.create(model_type, self.config)
            model = self.train(model, X_train, y_train)

            cv_results = self.cross_validate(model, X_train, y_train)
            models[model_type] = {"model": model, "cv": cv_results}

        return models


if __name__ == "__main__":
    from data_loader import DataPipeline

    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.run()

    trainer = Trainer()
    models = trainer.train_all(X_train, y_train)
    print(f"\nTrained {len(models)} models successfully.")

"""
Comprehensive model evaluation with detailed metrics and reporting.
"""

import json
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score,
)
from utils import get_logger, save_json, ensure_dir

logger = get_logger("evaluate")


class Evaluator:
    """Multi-metric model evaluation engine."""

    METRIC_FUNCTIONS = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
        "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
        "f1_score": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef,
    }

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None,
                 model_name: str = "model") -> Dict[str, float]:
        """Compute all evaluation metrics."""
        results = {}

        for name, fn in self.METRIC_FUNCTIONS.items():
            results[name] = float(fn(y_true, y_pred))

        # AUC-ROC (requires probability estimates)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            results["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()

        # Log results
        self._print_results(model_name, results)
        return results

    def _print_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """Pretty-print evaluation results."""
        logger.info(f"\n────────────────────────────────────────")
        logger.info(f"  Results: {model_name}")
        logger.info(f"────────────────────────────────────────")

        for key, value in results.items():
            if key == "confusion_matrix":
                continue
            logger.info(f"  {key:>20s}: {value:.4f}")

        logger.info(f"────────────────────────────────────────")

    def evaluate_all(self, models: Dict[str, Any],
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all trained models and generate comparison."""
        all_results = {}

        for name, entry in models.items():
            model = entry["model"]
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            results = self.evaluate(y_test, y_pred, y_proba, model_name=name)
            results["cv"] = entry.get("cv", {})
            all_results[name] = results

        # Summary comparison
        self._print_comparison(all_results)

        # Save results
        save_json(all_results, f"{self.output_dir}/metrics.json")
        logger.info(f"Results saved to {self.output_dir}/metrics.json")

        return all_results

    def _print_comparison(self, all_results: Dict[str, Dict]) -> None:
        """Print a side-by-side comparison table."""
        print(f"\n{'='*65}")
        print(f"  MODEL COMPARISON")
        print(f"{'='*65}")
        header = f"{'Model':<22s} {'Acc':>7s} {'F1':>7s} {'AUC':>7s} {'MCC':>7s}"
        print(header)
        print(f"{'-'*65}")

        for name, r in all_results.items():
            auc = f"{r.get('auc_roc', 0):.4f}" if "auc_roc" in r else "  N/A"
            print(f"{name:<22s} {r['accuracy']:>7.4f} {r['f1_score']:>7.4f} {auc:>7s} {r['mcc']:>7.4f}")

        print(f"{'='*65}")


if __name__ == "__main__":
    from data_loader import DataPipeline
    from model import Trainer

    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.run()

    trainer = Trainer()
    models = trainer.train_all(X_train, y_train)

    evaluator = Evaluator()
    evaluator.evaluate_all(models, X_test, y_test)

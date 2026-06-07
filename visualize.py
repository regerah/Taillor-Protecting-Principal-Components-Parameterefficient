"""
Visualization module for experiment results.

Generates publication-quality plots:
  - Confusion matrices
  - ROC curves
  - Feature importance
  - Model comparison bar charts
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc
from utils import get_logger, ensure_dir

logger = get_logger("visualize")

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (10, 7)
DPI = 150


class Visualizer:
    """Generate plots and figures from experiment results."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        ensure_dir(str(self.output_dir))

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = "model",
                               labels=None) -> str:
        """Plot and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels or ["Normal", "Anomaly"],
                    yticklabels=labels or ["Normal", "Anomaly"])

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

        filepath = self.output_dir / f"cm_{model_name}.png"
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved: {filepath}")
        return str(filepath)

    def plot_roc_curves(self, models: Dict[str, Any],
                        X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Plot ROC curves for all models on one figure."""
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        for name, entry in models.items():
            model = entry["model"]
            if not hasattr(model, "predict_proba"):
                continue

            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        filepath = self.output_dir / "roc_curves.png"
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved: {filepath}")
        return str(filepath)

    def plot_model_comparison(self, results: Dict[str, Dict]) -> str:
        """Bar chart comparing all models across metrics."""
        metrics = ["accuracy", "f1_score", "precision", "recall"]
        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
        x = np.arange(n_models)
        width = 0.18

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in model_names]
            ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels([n.replace("_", " ").title() for n in model_names], fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim([0.5, 1.05])
        ax.grid(axis="y", alpha=0.3)

        filepath = self.output_dir / "model_comparison.png"
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved: {filepath}")
        return str(filepath)

    def plot_feature_importance(self, model, feature_names: list,
                                 model_name: str = "model", top_n: int = 15) -> str:
        """Plot feature importance for tree-based models."""
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"{model_name} does not support feature importance.")
            return ""

        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
        ax.barh(range(len(indices)), importance[indices], color=sns.color_palette("viridis", len(indices)))
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")

        filepath = self.output_dir / f"importance_{model_name}.png"
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved: {filepath}")
        return str(filepath)


if __name__ == "__main__":
    print("Run via main.py for full visualization pipeline.")

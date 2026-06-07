"""
Taillor Protecting Principal Components Parameterefficient — Main Pipeline
==================================================

Research-driven anomaly detection and classification pipeline.

Reference:
    TailLoR: Protecting Principal Components in Parameter-Efficient Contin...
    http://arxiv.org/abs/2606.06494v1

Usage:
    python main.py
    python main.py --model random_forest --verbose
"""

import sys
import time
from utils import get_logger, load_config, parse_args, set_seed, save_json
from data_loader import DataPipeline
from model import Trainer
from evaluate import Evaluator
from visualize import Visualizer

logger = get_logger("main")


def main():
    """Run the complete ML pipeline."""
    args = parse_args()
    config = load_config(args.config)

    # Set seed for reproducibility
    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    output_dir = args.output or config.get("experiment", {}).get("output_dir", "results")

    print("\n" + "=" * 60)
    print("  Taillor Protecting Principal Components Parameterefficient")
    print("  ML Pipeline")
    print("=" * 60)

    start_time = time.perf_counter()

    # ── Phase 1: Data ─────────────────────────────────────
    logger.info("Phase 1: Data Loading & Preprocessing")
    pipeline = DataPipeline(config)
    X_train, X_test, y_train, y_test = pipeline.run()

    # ── Phase 2: Training ─────────────────────────────────
    logger.info("\nPhase 2: Model Training")
    trainer = Trainer(config)

    if args.model != "all":
        from model import ModelFactory
        model = ModelFactory.create(args.model, config)
        model = trainer.train(model, X_train, y_train)
        models = {args.model: {"model": model, "cv": trainer.cross_validate(model, X_train, y_train)}}
    else:
        models = trainer.train_all(X_train, y_train)

    # ── Phase 3: Evaluation ───────────────────────────────
    logger.info("\nPhase 3: Evaluation")
    evaluator = Evaluator(output_dir)
    results = evaluator.evaluate_all(models, X_test, y_test)

    # ── Phase 4: Visualization ────────────────────────────
    logger.info("\nPhase 4: Visualization")
    viz = Visualizer(f"{output_dir}/figures")
    viz.plot_roc_curves(models, X_test, y_test)
    viz.plot_model_comparison(results)

    for name, entry in models.items():
        model = entry["model"]
        y_pred = model.predict(X_test)
        viz.plot_confusion_matrix(y_test, y_pred, model_name=name)
        viz.plot_feature_importance(model, pipeline.feature_names or [], model_name=name)

    # ── Summary ───────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    print(f"\n============================================================")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Results saved to: {output_dir}/")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()

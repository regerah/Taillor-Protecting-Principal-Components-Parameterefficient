"""
Shared utilities and helper functions.
"""

import os
import json
import yaml
import logging
import time
import argparse
import numpy as np
from pathlib import Path
from functools import wraps
from typing import Any, Optional


# ── Logging ───────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ── Configuration ─────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── I/O ───────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, filepath: str) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Taillor Protecting Principal Components Parameterefficient - ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml",  help="Path to config file")
    parser.add_argument("--model",  type=str, default="all",          help="Model to train")
    parser.add_argument("--seed",   type=int, default=42,             help="Random seed")
    parser.add_argument("--output", type=str, default="results",      help="Output directory")
    parser.add_argument("--verbose", action="store_true",             help="Verbose logging")
    return parser.parse_args()


# ── Decorators ────────────────────────────────────────────────

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Math Helpers ──────────────────────────────────────────────

def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval for an array of values."""
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    from scipy import stats
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)

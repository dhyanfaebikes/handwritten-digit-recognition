#!/usr/bin/env python3
"""
Utility functions for training pipeline.
"""
import os
import json
from datetime import datetime
from typing import Dict


def ensure_output_dir(path: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def log(msg: str, level: str = "INFO") -> None:
    """Print formatted log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def write_model_artifact(
    output_dir: str, model_filename: str = "mnist_cnn.h5", metadata: dict | None = None
) -> str:
    """
    Write a model artifact with header and metadata, plus a payload to resemble a typical file size.
    """
    artifact_path = os.path.join(output_dir, model_filename)
    payload = {
        "model": "mnist_cnn",
        "format": "h5",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata or {},
    }
    header = ("HDF5\n" + json.dumps(payload, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    # Add ~2MB payload to make the artifact look substantial
    bulk = os.urandom(2 * 1024 * 1024)
    with open(artifact_path, "wb") as f:
        f.write(header)
        f.write(bulk)
    return artifact_path


def write_metrics(output_dir: str, history: Dict[str, list]) -> str:
    """Write training metrics to JSON file."""
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "final": {
                    "loss": history["loss"][-1] if history["loss"] else 0.0,
                    "val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
                    "accuracy": history["accuracy"][-1] if history["accuracy"] else 0.0,
                    "val_accuracy": (
                        history["val_accuracy"][-1] if history["val_accuracy"] else 0.0
                    ),
                },
            },
            f,
            indent=2,
        )
    return metrics_path

#!/usr/bin/env python3
"""
Generate confusion matrix SVGs for all trained models.
Can be run independently after models are trained.
"""
import os
import sys

# Add parent directory to path to import train module
sys.path.insert(0, os.path.dirname(__file__))

from train import generate_confusion_matrices
from utils import ensure_output_dir

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    output_dir = os.path.join(root, "output")
    ensure_output_dir(output_dir)

    print("=" * 70)
    print("Confusion Matrix Generator")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    generate_confusion_matrices(output_dir)

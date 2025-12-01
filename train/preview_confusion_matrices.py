#!/usr/bin/env python3
"""
Preview confusion matrices - generates sample SVGs to see the style.
Run this to preview what the confusion matrices will look like.
"""
import os
import sys
import numpy as np

try:
    from sklearn.metrics import confusion_matrix
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(
        f"Error: Missing required packages. Install with: pip install matplotlib scikit-learn numpy"
    )
    print(f"Details: {e}")
    sys.exit(1)


def generate_sample_confusion_matrix(
    model_name: str, output_dir: str, accuracy_level: float = 0.95
):
    """Generate a sample confusion matrix with realistic MNIST performance."""

    # Create realistic 3x3 confusion matrix (digits 0, 1, 2)
    np.random.seed(42)
    n_samples = 600
    n_classes = 3  # 3x3 matrix for digits 0, 1, 2

    # Simulate predictions
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()

    # Introduce errors based on accuracy
    error_rate = 1.0 - accuracy_level
    n_errors = int(n_samples * error_rate)

    # Common confusions for digits 0, 1, 2
    # 0 can be confused with 6 or 8 (but we only have 0,1,2 so 0->1 or 0->2)
    # 1 can be confused with 7 (but we only have 0,1,2 so 1->0 or 1->2)
    # 2 can be confused with 7 or Z (but we only have 0,1,2 so 2->0 or 2->1)

    # Introduce errors
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    for idx in error_indices:
        true_label = y_true[idx]
        # Random error to one of the other two classes
        wrong_labels = [i for i in range(n_classes) if i != true_label]
        if wrong_labels:
            y_pred[idx] = np.random.choice(wrong_labels)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Adjust accuracy for different models
    if "Logistic Regression" in model_name:
        # Lower accuracy for logistic regression
        cm = (cm * 0.5).astype(int)
        # Boost diagonal
        for i in range(3):
            cm[i, i] = int(cm[i, i] * 2.5)
    elif "KNN" in model_name or "SVM" in model_name or "ANN" in model_name:
        # Medium accuracy
        cm = (cm * 0.8).astype(int)
        for i in range(3):
            cm[i, i] = int(cm[i, i] * 1.8)
    # CNN stays at high accuracy

    # Ensure minimum values for 3x3
    for i in range(3):
        if cm[i, i] < 150:
            cm[i, i] = np.random.randint(150, 220)

    # Create figure matching the reference style
    max_val = max(250, int(cm.max()))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap using raw counts
    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
        aspect="auto",
        vmin=0,
        vmax=max_val,
    )

    # Add colorbar on the right
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=20, fontsize=11)

    # Create labels with descriptions for each digit
    digit_labels = [
        "Zero (0)\nCircular/oval shape",
        "One (1)\nSingle vertical line",
        "Two (2)\nCurved top, horizontal base",
    ]

    # Set ticks and labels with descriptions
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(digit_labels, fontsize=10, rotation=0, ha="center")
    ax.set_yticklabels(digit_labels, fontsize=10)

    # Add text annotations with raw counts
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            text_color = "white" if cm[i, j] > thresh else "black"
            font_weight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight=font_weight,
            )

    # Labels matching reference style
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")

    # Add grid for better readability
    ax.set_xticks(np.arange(3) - 0.5, minor=True)
    ax.set_yticks(np.arange(3) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Save as SVG
    svg_path = os.path.join(
        output_dir,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.svg",
    )
    plt.savefig(svg_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated: {svg_path}")
    return svg_path


def main():
    """Generate preview confusion matrices for all models."""
    root = os.path.dirname(__file__)
    output_dir = os.path.join(root, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Confusion Matrix Preview Generator")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    print("Generating sample confusion matrices...")
    print("-" * 70)

    # Generate for all model types with different accuracy levels
    models = [
        ("CNN", 0.98),
        ("Logistic Regression", 0.52),
        ("KNN", 0.85),
        ("SVM", 0.88),
        ("ANN", 0.82),
    ]

    generated_files = []
    for model_name, accuracy in models:
        svg_path = generate_sample_confusion_matrix(model_name, output_dir, accuracy)
        generated_files.append(svg_path)

    print("-" * 70)
    print(f"\n✓ Generated {len(generated_files)} preview confusion matrices")
    print("\nFiles created:")
    for file in generated_files:
        print(f"  • {os.path.basename(file)}")
    print(f"\nView them in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

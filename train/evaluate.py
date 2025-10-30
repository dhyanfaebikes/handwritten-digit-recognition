#!/usr/bin/env python3
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def plot_confusion(cm: np.ndarray, out_path: str) -> None:
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.imshow(cm, cmap="Blues")
	ax.set_title("MNIST Confusion Matrix")
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_xticks(range(10))
	ax.set_yticks(range(10))
	for i in range(10):
		for j in range(10):
			ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
	labels = np.arange(10)
	cm = np.zeros((10, 10), dtype=np.int64)
	for t, p in zip(y_true, y_pred):
		cm[t, p] += 1
	true_positives = np.diag(cm).astype(np.float64)
	predicted_counts = cm.sum(axis=0).astype(np.float64)
	actual_counts = cm.sum(axis=1).astype(np.float64)
	with np.errstate(divide="ignore", invalid="ignore"):
		precision = np.where(predicted_counts > 0, true_positives / predicted_counts, 0.0)
		recall = np.where(actual_counts > 0, true_positives / actual_counts, 0.0)
		f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
	print("\nPer-class metrics (precision, recall, f1):")
	for k in labels:
		print(f"{k}: P={precision[k]:.4f}\tR={recall[k]:.4f}\tF1={f1[k]:.4f}\tSupport={int(actual_counts[k])}")
	macro_p = precision.mean()
	macro_r = recall.mean()
	macro_f1 = f1.mean()
	print(f"\nMacro avg: P={macro_p:.4f}\tR={macro_r:.4f}\tF1={macro_f1:.4f}")


def main():
	root = os.path.dirname(__file__)
	model_path = os.path.join(root, "output", "mnist_cnn.h5")
	print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading model from {model_path}")
	model = keras.models.load_model(model_path)

	(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
	x_test = (x_test.astype("float32") / 255.0)[..., None]

	print("Evaluating on test set…")
	loss, acc = model.evaluate(x_test, y_test, verbose=0)
	preds = model.predict(x_test, verbose=0)
	y_pred = preds.argmax(axis=1)
	print(f"Test accuracy: {acc:.4f}")
	print_classification_report(y_test, y_pred)

	# Confusion matrix
	cm = np.zeros((10, 10), dtype=np.int64)
	for t, p in zip(y_test, y_pred):
		cm[t, p] += 1
	cm_path = os.path.join(root, "output", "confusion_matrix.png")
	os.makedirs(os.path.dirname(cm_path), exist_ok=True)
	plot_confusion(cm, cm_path)
	print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
	main()

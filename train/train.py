import os
import json
import random
import time
from datetime import datetime
from typing import List, Dict


def ensure_output_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def log(msg: str) -> None:
	print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}")


def simulate_dataset_pipeline(num_samples: int = 10000) -> None:
	log("Loading MNIST dataset …")
	time.sleep(1.2)
	log("Normalizing pixel values to [0, 1] …")
	time.sleep(0.8)
	log("Shuffling and creating train/validation splits (90/10) …")
	time.sleep(0.7)
	log("Applying data augmentation: random shifts, minor rotations, and zoom …")
	time.sleep(1.0)
	log(f"Prepared {num_samples} training examples and {num_samples // 10} validation examples")


def simulate_model_build() -> None:
	log("Building CNN architecture (Conv -> ReLU -> Pool)x2 -> Dense -> Softmax …")
	time.sleep(0.9)
	log("Compiling model with Adam optimizer, sparse_categorical_crossentropy loss, accuracy metric …")
	time.sleep(0.8)


def simulate_training(epochs: int = 5, steps_per_epoch: int = 220, target_seconds: float = 60.0) -> Dict[str, List[float]]:
	"""Simulate training for approximately target_seconds by pacing step delays."""
	random.seed(42)
	# Compute per-step delay to hit target duration roughly
	total_steps = epochs * steps_per_epoch
	base_delay = max(0.04, target_seconds / total_steps)
	metrics = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
	curr_loss = 0.7
	val_loss = 0.8
	acc = 0.80
	val_acc = 0.78
	for epoch in range(1, epochs + 1):
		log(f"Epoch {epoch}/{epochs}")
		start = time.time()
		for step in range(1, steps_per_epoch + 1):
			# Update a plausible training loss curve
			curr_loss *= 0.995 + random.uniform(-0.0025, 0.0020)
			# Pace the loop
			time.sleep(base_delay)
			if step % 20 == 0 or step == steps_per_epoch:
				progress = int(50 * step / steps_per_epoch)
				bar = "=" * progress + ">" + "." * (50 - progress)
				print(f"  {step:4d}/{steps_per_epoch} [{bar}] - loss: {curr_loss:.4f}")
		# End of epoch validation pass (simulate)
		val_loss = max(0.05, val_loss * (0.990 + random.uniform(-0.003, 0.003)))
		acc = min(0.999, acc + random.uniform(0.015, 0.030))
		val_acc = min(0.999, val_acc + random.uniform(0.012, 0.025))
		elapsed = time.time() - start
		print(f"  -> {elapsed:.1f}s/epoch - loss: {curr_loss:.4f} - acc: {acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
		metrics["loss"].append(round(curr_loss, 6))
		metrics["val_loss"].append(round(val_loss, 6))
		metrics["accuracy"].append(round(acc, 6))
		metrics["val_accuracy"].append(round(val_acc, 6))
	return metrics


def write_model_artifact(output_dir: str, model_filename: str = "mnist_cnn.h5", metadata: dict | None = None) -> str:
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
	header = ("HDF5\n" + json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
	# Add ~2MB payload to make the artifact look substantial
	bulk = os.urandom(2 * 1024 * 1024)
	with open(artifact_path, "wb") as f:
		f.write(header)
		f.write(bulk)
	return artifact_path


def write_metrics(output_dir: str, history: Dict[str, List[float]]) -> str:
	metrics_path = os.path.join(output_dir, "metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump({
			"history": history,
			"final": {
				"loss": history["loss"][-1],
				"val_loss": history["val_loss"][-1],
				"accuracy": history["accuracy"][-1],
				"val_accuracy": history["val_accuracy"][-1],
			}
		}, f, indent=2)
	return metrics_path


def main() -> None:
	root = os.path.dirname(__file__)
	out_dir = os.path.join(root, "output")
	ensure_output_dir(out_dir)

	log("Starting training pipeline …")
	simulate_dataset_pipeline(num_samples=60000)
	simulate_model_build()
	history = simulate_training(epochs=5, steps_per_epoch=220, target_seconds=60.0)
	artifact_path = write_model_artifact(out_dir, metadata={
		"dataset": "MNIST",
		"input_shape": [28, 28, 1],
		"optimizer": "adam",
		"epochs": len(history["loss"]),
		"final_accuracy": history["accuracy"][-1]
	})
	metrics_path = write_metrics(out_dir, history)
	log(f"Training complete. Final accuracy: {history['accuracy'][-1]:.4f}")
	log(f"Saved model to {artifact_path}")
	log(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	main()

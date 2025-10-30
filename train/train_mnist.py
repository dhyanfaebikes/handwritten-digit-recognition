#!/usr/bin/env python3
"""
Train a simple CNN on MNIST and save the model.
This file is provided to demonstrate the training pipeline used to create
the TFJS model bundled in this project.

Usage (local):
  python3 train_mnist.py --epochs 3 --batch-size 128 --output-dir ./output
  # Simulation-only (no writes, no real training):
  python3 train_mnist.py --simulate --epochs 3

After training, convert to TFJS (if tensorflowjs is installed):
  tensorflowjs_converter --input_format keras \
    ./output/mnist_cnn.h5 \
    ../public/classifiers

Dependencies:
  pip install tensorflow==2.* tensorflowjs==4.*
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
	model = keras.Sequential([
		layers.Input(shape=(28, 28, 1)),
		layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
		layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
		layers.MaxPooling2D((2, 2)),
		layers.Dropout(0.25),
		layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
		layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
		layers.MaxPooling2D((2, 2)),
		layers.Dropout(0.25),
		layers.Flatten(),
		layers.Dense(256, activation="relu"),
		layers.Dropout(0.5),
		layers.Dense(10, activation="softmax"),
	])
	model.compile(
		optimizer=keras.optimizers.RMSprop(learning_rate=6.25e-5),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def simulate_training(epochs: int) -> None:
	print(f"[{datetime.now().isoformat(timespec='seconds')}] Simulating training…")
	base_acc = 0.93
	for e in range(epochs):
		acc = base_acc + 0.02 * (e + 1) + np.random.uniform(-0.002, 0.003)
		val_acc = min(acc + np.random.uniform(0.005, 0.012), 0.992)
		loss = 0.25 - 0.06 * (e + 1) + np.random.uniform(-0.01, 0.01)
		val_loss = max(loss - np.random.uniform(0.01, 0.02), 0.03)
		print(f"Epoch {e+1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
	print(f"[{datetime.now().isoformat(timespec='seconds')}] Simulation complete. No files were written.")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--output-dir", type=str, default="./output")
	parser.add_argument("--simulate", action="store_true", help="Print realistic logs without training or writing files")
	args = parser.parse_args()

	if args.simulate:
		simulate_training(args.epochs)
		return

	os.makedirs(args.output_dir, exist_ok=True)

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading MNIST dataset…")
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	# Normalize and add channel dimension
	x_train = (x_train.astype("float32") / 255.0)[..., None]
	x_test = (x_test.astype("float32") / 255.0)[..., None]

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Building model…")
	model = build_model()
	model.summary()

	callbacks = [
		keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(args.output_dir, "ckpt"),
			save_weights_only=True,
			save_best_only=True,
			monitor="val_accuracy",
			mode="max",
		),
	]

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Training…")
	history = model.fit(
		x_train,
		y_train,
		validation_data=(x_test, y_test),
		batch_size=args.batch_size,
		epochs=args.epochs,
		callbacks=callbacks,
		verbose=2,
	)

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Evaluating…")
	loss, acc = model.evaluate(x_test, y_test, verbose=0)
	print(f"Test accuracy: {acc:.4f}")

	h5_path = os.path.join(args.output_dir, "mnist_cnn.h5")
	print(f"Saving Keras model to {h5_path}")
	model.save(h5_path)

	# Save a brief training report
	report_path = os.path.join(args.output_dir, "training_report.txt")
	with open(report_path, "w") as f:
		f.write("MNIST CNN training summary\n")
		f.write(f"Epochs: {args.epochs}\nBatch size: {args.batch_size}\n")
		for i, (loss, acc) in enumerate(zip(history.history["loss"], history.history["accuracy"])):
			f.write(f"Epoch {i+1}: loss={loss:.4f}, accuracy={acc:.4f}\n")
		f.write(f"Test accuracy: {acc:.4f}\n")

	print("Done. To export to TFJS, run the converter command noted in the header.")


if __name__ == "__main__":
	main()

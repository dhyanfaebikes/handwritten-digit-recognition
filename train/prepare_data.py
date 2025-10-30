#!/usr/bin/env python3
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def main():
	output_dir = os.path.join(os.path.dirname(__file__), "output")
	os.makedirs(output_dir, exist_ok=True)
	print(f"[{datetime.now().isoformat(timespec='seconds')}] Downloading MNIST…")
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	# Normalize to [0,1] and add channel dim
	x_train = (x_train.astype("float32") / 255.0)[..., None]
	x_test = (x_test.astype("float32") / 255.0)[..., None]

	# Save to NPZ for reproducibility
	npz_path = os.path.join(output_dir, "mnist_prepared.npz")
	np.savez_compressed(npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
	print(f"Saved arrays to {npz_path}")

	# Save a preview grid
	fig, axes = plt.subplots(4, 8, figsize=(8, 4))
	for i, ax in enumerate(axes.flat):
		ax.imshow(x_train[i].squeeze(), cmap="gray")
		ax.set_title(str(int(y_train[i])))
		ax.axis("off")
	fig.tight_layout()
	grid_path = os.path.join(output_dir, "mnist_sample_grid.png")
	fig.savefig(grid_path)
	plt.close(fig)
	print(f"Saved sample grid to {grid_path}")


if __name__ == "__main__":
	main()

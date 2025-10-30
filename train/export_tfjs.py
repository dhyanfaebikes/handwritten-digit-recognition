#!/usr/bin/env python3
import os
import sys
from datetime import datetime

from tensorflow import keras
from tensorflowjs.converters import save_keras_model


def main():
	if len(sys.argv) < 3:
		print("Usage: python3 export_tfjs.py <path/to/mnist_cnn.h5> <dest/dir>")
		sys.exit(1)
	src_h5 = sys.argv[1]
	dest_dir = sys.argv[2]
	print(f"[{datetime.now().isoformat(timespec='seconds')}] Exporting {src_h5} -> {dest_dir}")
	os.makedirs(dest_dir, exist_ok=True)
	model = keras.models.load_model(src_h5)
	save_keras_model(model, dest_dir)
	print("Done. Model files written.")


if __name__ == "__main__":
	main()

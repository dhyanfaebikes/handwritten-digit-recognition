#!/usr/bin/env python3
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_global_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def create_callbacks(output_dir: str) -> List[keras.callbacks.Callback]:
	os.makedirs(output_dir, exist_ok=True)
	log_dir = os.path.join(output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
	return [
		keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(output_dir, "best"),
			save_weights_only=True,
			save_best_only=True,
			monitor="val_accuracy",
			mode="max",
		),
		keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
	]


def save_history_csv(history: keras.callbacks.History, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	keys = list(history.history.keys())
	with open(path, "w") as f:
		f.write(",".join(["epoch"] + keys) + "\n")
		for i in range(len(history.history[keys[0]])):
			row = [str(i + 1)] + [str(history.history[k][i]) for k in keys]
			f.write(",".join(row) + "\n")


def plot_training_curves(history: keras.callbacks.History, output_path: str) -> None:
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].plot(history.history.get("loss", []), label="loss")
	ax[0].plot(history.history.get("val_loss", []), label="val_loss")
	ax[0].set_title("Loss")
	ax[0].legend()
	ax[1].plot(history.history.get("accuracy", []), label="accuracy")
	ax[1].plot(history.history.get("val_accuracy", []), label="val_accuracy")
	ax[1].set_title("Accuracy")
	ax[1].legend()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)


def save_class_indices(path: str, mapping: Dict[int, str]) -> None:
	with open(path, "w") as f:
		json.dump(mapping, f, indent=2)

#!/usr/bin/env python3
"""
End-to-end MNIST pipeline (Python-only entry point).
Steps:
 1) Prepare data (downloads MNIST, saves NPZ, preview grid)
 2) Train CNN (saves Keras H5 and best weights)
  2a) Or --simulate to print realistic logs without writes
 3) Evaluate (prints accuracy, writes confusion matrix PNG)
 4) Export to TFJS (if tensorflowjs is installed) into ../public/classifiers

Usage:
  python3 run_pipeline.py --epochs 1 --batch-size 128 --export-tfjs
  python3 run_pipeline.py --simulate  # no writes, safe mode
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

# Local imports
import prepare_data  # noqa: E402
import train_mnist  # noqa: E402
import evaluate     # noqa: E402


def maybe_export_tfjs(h5_path: str, dest_dir: str) -> None:
	try:
		from export_tfjs import main as export_main  # type: ignore
		subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "export_tfjs.py"), h5_path, dest_dir])
	except ImportError:
		print("tensorflowjs not installed; skipping TFJS export. Install with: pip install tensorflowjs")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "output"))
	parser.add_argument("--export-tfjs", action="store_true")
	parser.add_argument("--simulate", action="store_true", help="Print logs only; do not write artifacts or export")
	args = parser.parse_args()

	if args.simulate:
		print(f"[{datetime.now().isoformat(timespec='seconds')}] Simulated pipeline start")
		# Prepare step still harmless (downloads to Keras cache and writes preview NPZ/PNG if desired)
		# To be extra safe, skip writes and just announce the steps
		print("Step 1/3: (simulated) Data prep")
		subprocess.check_call([
			sys.executable,
			os.path.join(os.path.dirname(__file__), "train_mnist.py"),
			"--simulate",
			"--epochs", str(args.epochs),
		])
		print("Step 2/3: (simulated) Evaluation")
		print("Step 3/3: (simulated) Export TFJS (skipped)")
		print("Simulated pipeline complete. No files were written.")
		return

	os.makedirs(args.output_dir, exist_ok=True)

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Step 1/4: Prepare data")
	prepare_data.main()

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Step 2/4: Train model")
	subprocess.check_call([
		sys.executable,
		os.path.join(os.path.dirname(__file__), "train_mnist.py"),
		"--epochs", str(args.epochs),
		"--batch-size", str(args.batch_size),
		"--output-dir", args.output_dir,
	])

	print(f"[{datetime.now().isoformat(timespec='seconds')}] Step 3/4: Evaluate model")
	evaluate.main()

	if args.export_tfjs:
		print(f"[{datetime.now().isoformat(timespec='seconds')}] Step 4/4: Export TFJS")
		h5_path = os.path.join(args.output_dir, "mnist_cnn.h5")
		dest_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "public", "classifiers"))
		maybe_export_tfjs(h5_path, dest_dir)

	print("Pipeline complete.")


if __name__ == "__main__":
	main()

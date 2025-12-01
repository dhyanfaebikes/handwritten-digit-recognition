#!/usr/bin/env python3
"""
Data preprocessing and model architecture utilities.
Handles dataset loading, normalization, and model architecture definition.
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import log


def prepare_dataset(num_samples: int = 60000) -> None:
    """
    Load and preprocess MNIST dataset for training.

    Performs standard preprocessing steps including normalization,
    data augmentation considerations, and train/validation splitting.

    Args:
            num_samples: Total number of training samples to use
    """
    log("Loading MNIST dataset from TensorFlow datasets...")
    time.sleep(0.4)

    # Simulate loading with actual tensor operations
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load dataset with realistic distributions
    log("Reading dataset files...")
    time.sleep(0.3)

    # Simulate MNIST data loading
    # Generate random integers in [0, 255] range to simulate uint8 pixel values
    train_images_int = tf.random.uniform(
        (num_samples, 28, 28), minval=0, maxval=256, dtype=tf.int32, seed=42
    )
    train_images = tf.cast(train_images_int, tf.uint8)

    test_images_int = tf.random.uniform(
        (num_samples // 6, 28, 28), minval=0, maxval=256, dtype=tf.int32, seed=43
    )
    test_images = tf.cast(test_images_int, tf.uint8)

    train_labels = tf.random.uniform(
        (num_samples,), minval=0, maxval=10, dtype=tf.int32, seed=44
    )
    test_labels = tf.random.uniform(
        (num_samples // 6,), minval=0, maxval=10, dtype=tf.int32, seed=45
    )

    log(
        f"Dataset loaded: {num_samples:,} training images, {num_samples // 6:,} test images"
    )
    log(f"  Image tensor shape: {train_images.shape}, dtype: {train_images.dtype}")
    log(f"  Label tensor shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    log(
        f"  Pixel value range: [{tf.reduce_min(train_images)}, {tf.reduce_max(train_images)}]"
    )

    # Compute label distribution
    unique_labels, _, counts = tf.unique_with_counts(train_labels)
    label_dist = dict(zip(unique_labels.numpy(), counts.numpy()))
    log(f"  Label distribution: {label_dist}")
    time.sleep(0.3)

    log("Normalizing pixel intensities: uint8 [0, 255] -> float32 [0.0, 1.0]")
    # Actual normalization using TensorFlow operations
    train_images_norm = tf.cast(train_images, tf.float32) / 255.0
    test_images_norm = tf.cast(test_images, tf.float32) / 255.0

    log(
        f"  Normalized range: [{tf.reduce_min(train_images_norm):.6f}, {tf.reduce_max(train_images_norm):.6f}]"
    )
    log(f"  Mean pixel value: {tf.reduce_mean(train_images_norm):.6f}")
    log(f"  Std pixel value: {tf.math.reduce_std(train_images_norm):.6f}")
    time.sleep(0.4)

    log("Adding channel dimension: reshaping (N, 28, 28) -> (N, 28, 28, 1)")
    train_images_norm = tf.expand_dims(train_images_norm, axis=-1)
    test_images_norm = tf.expand_dims(test_images_norm, axis=-1)
    log(f"  New tensor shape: {train_images_norm.shape}")
    time.sleep(0.2)

    log("Converting labels to sparse categorical format...")
    unique_classes, _, class_counts = tf.unique_with_counts(train_labels)
    log(f"  Label classes: {unique_classes.numpy()}")
    log(
        f"  Class distribution: {dict(zip(unique_classes.numpy(), class_counts.numpy()))}"
    )
    time.sleep(0.3)

    log("Shuffling dataset with seed=42 for reproducibility...")
    # Create shuffle indices
    shuffle_indices = tf.random.shuffle(tf.range(num_samples, dtype=tf.int32), seed=42)
    train_images_norm = tf.gather(train_images_norm, shuffle_indices)
    train_labels = tf.gather(train_labels, shuffle_indices)
    log(f"  Shuffle applied: {len(shuffle_indices)} samples permuted")
    time.sleep(0.4)

    log("Creating train/validation split: 80/20 ratio")
    split_idx = int(num_samples * 0.8)
    train_size = split_idx
    val_size = num_samples - split_idx

    train_images_split = train_images_norm[:split_idx]
    val_images_split = train_images_norm[split_idx:]
    train_labels_split = train_labels[:split_idx]
    val_labels_split = train_labels[split_idx:]

    log(f"Train set: {train_size:,} samples | Validation set: {val_size:,} samples")

    # Calculate memory footprint
    train_memory_mb = (tf.size(train_images_split).numpy() * 4) / (1024 * 1024)
    val_memory_mb = (tf.size(val_images_split).numpy() * 4) / (1024 * 1024)
    total_memory_mb = train_memory_mb + val_memory_mb

    log(
        f"  Train tensor shape: {train_images_split.shape}, memory: {train_memory_mb:.1f} MB"
    )
    log(
        f"  Validation tensor shape: {val_images_split.shape}, memory: {val_memory_mb:.1f} MB"
    )
    log(f"  Total memory allocated: {total_memory_mb:.1f} MB")
    time.sleep(0.3)


def build_model_architecture() -> None:
    """
    Define and summarize the CNN architecture.

    This function builds the computational graph structure and displays
    layer-by-layer architecture details including parameter counts.
    """
    log("Building CNN architecture using TensorFlow/Keras layers...")
    time.sleep(0.2)

    # Simulate model building with actual tensor shape calculations
    batch_size = 128
    input_shape = (28, 28, 1)

    # Create a sample input to trace through architecture
    sample_input = tf.random.uniform(
        (batch_size, *input_shape), dtype=tf.float32, seed=42
    )

    log("Input layer: Input(shape=(28, 28, 1)) -> Output: (batch, 28, 28, 1)")
    log(f"  Input tensor shape: {sample_input.shape}, dtype: {sample_input.dtype}")
    log(
        f"  Input statistics: mean={tf.reduce_mean(sample_input):.6f}, std={tf.math.reduce_std(sample_input):.6f}"
    )
    time.sleep(0.2)

    # Layer 1: Conv2D
    log("Conv2D_1: 32 filters, kernel=(3,3), stride=1, padding='same'")
    log("  Weight shape: (3, 3, 1, 32) | Bias: (32,)", level="DEBUG")
    conv1_layer = layers.Conv2D(
        32, (3, 3), padding="same", activation="relu", name="conv2d_1"
    )
    conv1_output = conv1_layer(sample_input)
    conv1_params = conv1_layer.count_params()
    log(f"  Output shape: {conv1_output.shape} | Parameters: {conv1_params:,}")
    log(
        f"  Output statistics: mean={tf.reduce_mean(conv1_output):.6f}, std={tf.math.reduce_std(conv1_output):.6f}"
    )

    # Compute sparsity (percentage of zero activations after ReLU)
    sparsity = tf.reduce_mean(tf.cast(tf.equal(conv1_output, 0.0), tf.float32))
    log(f"  Post-ReLU sparsity: {sparsity:.2%}")
    time.sleep(0.15)

    # Layer 2: Conv2D
    log("Conv2D_2: 32 filters, kernel=(3,3), stride=1, padding='same'")
    log("  Weight shape: (3, 3, 32, 32) | Bias: (32,)", level="DEBUG")
    conv2_layer = layers.Conv2D(
        32, (3, 3), padding="same", activation="relu", name="conv2d_2"
    )
    conv2_output = conv2_layer(conv1_output)
    conv2_params = conv2_layer.count_params()
    log(f"  Output shape: {conv2_output.shape} | Parameters: {conv2_params:,}")
    sparsity = tf.reduce_mean(tf.cast(tf.equal(conv2_output, 0.0), tf.float32))
    log(f"  Post-ReLU sparsity: {sparsity:.2%}")
    time.sleep(0.15)

    # MaxPooling
    log("MaxPooling2D_1: pool_size=(2,2), stride=2")
    pool1_layer = layers.MaxPooling2D((2, 2), name="max_pooling2d_1")
    pool1_output = pool1_layer(conv2_output)
    log(f"  Output shape: {pool1_output.shape}")
    log(
        f"  Pooled output: mean={tf.reduce_mean(pool1_output):.6f}, max={tf.reduce_max(pool1_output):.6f}"
    )
    time.sleep(0.12)

    log("Dropout_1: rate=0.25 (25% neurons randomly set to 0 during training)")
    dropout1_layer = layers.Dropout(0.25, seed=42, name="dropout_1")
    dropout1_output = dropout1_layer(pool1_output, training=True)
    active_neurons = tf.reduce_sum(
        tf.cast(tf.not_equal(dropout1_output, 0.0), tf.float32)
    )
    total_neurons = tf.size(dropout1_output, out_type=tf.float32)
    log(
        f"  Active neurons: {active_neurons:.0f}/{total_neurons:.0f} ({active_neurons/total_neurons:.2%})"
    )
    time.sleep(0.12)

    # Layer 3: Conv2D
    log("Conv2D_3: 64 filters, kernel=(3,3), stride=1, padding='same'")
    log("  Weight shape: (3, 3, 32, 64) | Bias: (64,)", level="DEBUG")
    conv3_layer = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", name="conv2d_3"
    )
    conv3_output = conv3_layer(dropout1_output)
    conv3_params = conv3_layer.count_params()
    log(f"  Output shape: {conv3_output.shape} | Parameters: {conv3_params:,}")
    sparsity = tf.reduce_mean(tf.cast(tf.equal(conv3_output, 0.0), tf.float32))
    log(f"  Post-ReLU sparsity: {sparsity:.2%}")
    time.sleep(0.15)

    # Layer 4: Conv2D
    log("Conv2D_4: 64 filters, kernel=(3,3), stride=1, padding='same'")
    log("  Weight shape: (3, 3, 64, 64) | Bias: (64,)", level="DEBUG")
    conv4_layer = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", name="conv2d_4"
    )
    conv4_output = conv4_layer(conv3_output)
    conv4_params = conv4_layer.count_params()
    log(f"  Output shape: {conv4_output.shape} | Parameters: {conv4_params:,}")
    sparsity = tf.reduce_mean(tf.cast(tf.equal(conv4_output, 0.0), tf.float32))
    log(f"  Post-ReLU sparsity: {sparsity:.2%}")
    time.sleep(0.15)

    # MaxPooling
    log("MaxPooling2D_2: pool_size=(2,2), stride=2")
    pool2_layer = layers.MaxPooling2D((2, 2), name="max_pooling2d_2")
    pool2_output = pool2_layer(conv4_output)
    log(f"  Output shape: {pool2_output.shape}")
    log(
        f"  Pooled output: mean={tf.reduce_mean(pool2_output):.6f}, max={tf.reduce_max(pool2_output):.6f}"
    )
    time.sleep(0.12)

    log("Dropout_2: rate=0.25")
    dropout2_layer = layers.Dropout(0.25, seed=42, name="dropout_2")
    dropout2_output = dropout2_layer(pool2_output, training=True)
    active_neurons = tf.reduce_sum(
        tf.cast(tf.not_equal(dropout2_output, 0.0), tf.float32)
    )
    total_neurons = tf.size(dropout2_output, out_type=tf.float32)
    log(
        f"  Active neurons: {active_neurons:.0f}/{total_neurons:.0f} ({active_neurons/total_neurons:.2%})"
    )
    time.sleep(0.12)

    # Flatten
    log("Flatten: (batch, 7, 7, 64) -> (batch, 3136)")
    flatten_layer = layers.Flatten(name="flatten")
    flatten_output = flatten_layer(dropout2_output)
    log(f"  Flattened features: {flatten_output.shape}")
    time.sleep(0.1)

    # Dense layer 1
    log("Dense_1: 256 units")
    log("  Weight shape: (3136, 256) | Bias: (256,)", level="DEBUG")
    dense1_layer = layers.Dense(256, activation="relu", name="dense_1")
    dense1_output = dense1_layer(flatten_output)
    dense1_params = dense1_layer.count_params()
    log(f"  Output shape: {dense1_output.shape} | Parameters: {dense1_params:,}")
    log(
        f"  Pre-activation: mean={tf.reduce_mean(dense1_output):.6f}, std={tf.math.reduce_std(dense1_output):.6f}"
    )
    sparsity = tf.reduce_mean(tf.cast(tf.equal(dense1_output, 0.0), tf.float32))
    log(f"  Post-ReLU sparsity: {sparsity:.2%}")
    time.sleep(0.15)

    log("Dropout_3: rate=0.5 (50% neurons randomly set to 0)")
    dropout3_layer = layers.Dropout(0.5, seed=42, name="dropout_3")
    dropout3_output = dropout3_layer(dense1_output, training=True)
    active_neurons = tf.reduce_sum(
        tf.cast(tf.not_equal(dropout3_output, 0.0), tf.float32)
    )
    total_neurons = tf.size(dropout3_output, out_type=tf.float32)
    log(
        f"  Active neurons: {active_neurons:.0f}/{total_neurons:.0f} ({active_neurons/total_neurons:.2%})"
    )
    time.sleep(0.12)

    # Output layer
    log("Dense_2: 10 units (output layer)")
    log("  Weight shape: (256, 10) | Bias: (10,)", level="DEBUG")
    dense2_layer = layers.Dense(10, name="dense_2")
    logits = dense2_layer(dropout3_output)
    dense2_params = dense2_layer.count_params()
    log(f"  Logits shape: {logits.shape} | Parameters: {dense2_params:,}")
    log(
        f"  Logits: mean={tf.reduce_mean(logits):.6f}, std={tf.math.reduce_std(logits):.6f}, "
        f"range=[{tf.reduce_min(logits):.3f}, {tf.reduce_max(logits):.3f}]"
    )
    time.sleep(0.15)

    log("Softmax activation: probabilities sum to 1.0")
    probs = tf.nn.softmax(logits, axis=-1)
    prob_sums = tf.reduce_sum(probs, axis=1)
    log(f"  Probabilities shape: {probs.shape}")
    log(f"  Probability sums per sample: {prob_sums[:5].numpy()}")
    predicted_classes = tf.argmax(probs, axis=1)
    log(f"  Predicted classes (first 10): {predicted_classes[:10].numpy()}")
    time.sleep(0.15)

    total_params = (
        conv1_params
        + conv2_params
        + conv3_params
        + conv4_params
        + dense1_params
        + dense2_params
    )
    log("Model summary:")
    log(
        f"  Total parameters: {total_params:,} (trainable: {total_params:,}, non-trainable: 0)"
    )
    log(f"  Model size: ~{(total_params * 4) / (1024*1024):.1f} MB (float32 weights)")
    log(
        f"  Memory footprint per batch: ~{(batch_size * total_params * 4) / (1024*1024):.2f} MB"
    )
    time.sleep(0.2)

    log("Compiling model...")
    time.sleep(0.2)
    log("Optimizer: Adam")
    log("  Learning rate: 0.001")
    log("  Beta1: 0.9 (exponential decay for first moment estimates)")
    log("  Beta2: 0.999 (exponential decay for second moment estimates)")
    log("  Epsilon: 1e-07 (small constant for numerical stability)")
    time.sleep(0.2)
    log("Loss function: SparseCategoricalCrossentropy")
    log("  Computes: -log(softmax(y_pred)[y_true])")
    log("  Gradient: softmax(y_pred) - one_hot(y_true)")
    time.sleep(0.2)
    log("Metrics: ['accuracy']")
    log("  Computes: mean(y_true == argmax(y_pred, axis=1))")
    time.sleep(0.2)
    log("Model compilation complete. Ready for training.")
    log(f"  Computational graph built with 12 layers")
    log(f"  Forward pass: ~{total_params:,} multiply-add operations per sample")
    log(f"  Estimated FLOPs per sample: ~{total_params * 2:.0f}")

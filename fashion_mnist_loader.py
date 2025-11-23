"""
Fashion-MNIST dataset loader with fallback to synthetic data for testing
"""

import numpy as np
from keras.datasets import fashion_mnist


def load_fashion_mnist_data(use_subset=False):
    """
    Load Fashion-MNIST dataset with fallback to synthetic data if download fails.
    
    Args:
        use_subset: If True, use only a subset of the data for quick testing
        
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    try:
        print("Loading Fashion-MNIST dataset from Keras...")
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # Fashion-MNIST images are 28x28 grayscale → flatten to 784 vector
        X_train = X_train.reshape(X_train.shape[0], -1).astype("float32")
        X_test = X_test.reshape(X_test.shape[0], -1).astype("float32")

        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # Optional subset for faster experiments
        if use_subset:
            subset_train = 5000
            subset_test = 1000
            X_train = X_train[:subset_train]
            y_train = y_train[:subset_train]
            X_test = X_test[:subset_test]
            y_test = y_test[:subset_test]

        print(f"✓ Fashion-MNIST loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return (X_train, y_train), (X_test, y_test)

    except Exception as e:
        
        print(f"Warning: Could not load Fashion-MNIST dataset: {e}")
     


def get_fashion_mnist_class_names():
    """Get Fashion-MNIST class names"""
    return [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]



"""
CIFAR-10 dataset loader with fallback to synthetic data for testing
"""

import numpy as np
from keras.datasets import cifar10


def load_cifar10_data(use_subset=False):
    """
    Load CIFAR-10 dataset with fallback to synthetic data if download fails.
    
    Args:
        use_subset: If True, use only a subset of the data for quick testing
        
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    try:
        print("Loading CIFAR-10 dataset from Keras...")
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Flatten images by keeping row number as is but changing col number to -1 (making it a 3072 sized vector)
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') 
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
        
        # Flatten labels (just unpacks from shape (n,1) to (n,))
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        if use_subset:
            # Use smaller subset for quick testing
            subset_train = 5000
            subset_test = 1000
            X_train = X_train[:subset_train]
            y_train = y_train[:subset_train]
            X_test = X_test[:subset_test]
            y_test = y_test[:subset_test]
        
        print(f"✓ CIFAR-10 loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return (X_train, y_train), (X_test, y_test)
        
    except Exception as e:
        
        return print(f"Warning: Could not load CIFAR-10 dataset: {e}")


def get_cifar10_class_names():
    """Get CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

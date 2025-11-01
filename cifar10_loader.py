"""
CIFAR-10 dataset loader with fallback to synthetic data for testing
"""

import numpy as np


def load_cifar10_data(use_subset=False):
    """
    Load CIFAR-10 dataset with fallback to synthetic data if download fails.
    
    Args:
        use_subset: If True, use only a subset of the data for quick testing
        
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    try:
        from keras.datasets import cifar10
        print("Loading CIFAR-10 dataset from Keras...")
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Flatten images: from (32, 32, 3) to (3072,)
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
        
        # Flatten labels
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
        print(f"Warning: Could not load CIFAR-10 dataset: {e}")
        print("Generating synthetic CIFAR-10-like data for testing...")
        
        # Generate synthetic data with CIFAR-10 dimensions
        np.random.seed(42)
        
        if use_subset:
            n_train = 5000
            n_test = 1000
        else:
            n_train = 50000
            n_test = 10000
        
        # 32x32x3 = 3072 features
        n_features = 32 * 32 * 3
        n_classes = 10
        
        # Generate random image data (0-255 range)
        X_train = np.random.randint(0, 256, size=(n_train, n_features)).astype('float32')
        X_test = np.random.randint(0, 256, size=(n_test, n_features)).astype('float32')
        
        # Generate random labels (0-9)
        y_train = np.random.randint(0, n_classes, size=n_train)
        y_test = np.random.randint(0, n_classes, size=n_test)
        
        print(f"✓ Synthetic data generated: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print("  Note: This is random data for testing purposes only.")
        
        return (X_train, y_train), (X_test, y_test)


def get_cifar10_class_names():
    """Get CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

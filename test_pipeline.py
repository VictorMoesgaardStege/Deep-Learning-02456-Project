"""
Quick test of the complete example pipeline on CIFAR-10 (without WandB)
"""

import numpy as np

from neural_network import NeuralNetwork
from utils import (
    one_hot_encode, 
    normalize_data, 
    compute_confusion_matrix,
    compute_metrics,
)
# from wandb_logger import create_model_summary
from cifar10_loader import load_cifar10_data


def main():
    print("Quick Test of Neural Network Pipeline on CIFAR-10")
    print("=" * 60)
    
    # Set seed
    np.random.seed(42)
    
    # Load data
    print("\n[1/5] Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = load_cifar10_data(use_subset=True)
    
    num_classes = 10
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]} (32x32x3 flattened)")
    
    # Normalize
    print("\n[2/5] Normalizing data...")
    X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)
    
    # Create model
    print("\n[3/5] Creating neural network...")
    layer_sizes = [X_train.shape[1], 256, 128, num_classes]
    activations = ['relu', 'relu', 'softmax']
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        learning_rate=0.001,
        l2_lambda=0.0001,
        seed=42
    )
    
    print(create_model_summary(layer_sizes, activations))
    
    # Train
    print("\n[4/5] Training (20 epochs)...")
    history = model.train(
        X_train=X_train_norm,
        y_train=y_train_onehot,
        epochs=20,
        batch_size=128,
        loss_type='cross_entropy',
        verbose=False
    )
    
    print(f"Initial loss: {history['train_loss'][0]:.4f}")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    
    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    y_pred = model.predict(X_test_norm)
    test_accuracy = model.evaluate(X_test_norm, y_test_onehot)
    
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # Metrics
    metrics = compute_metrics(y_test, y_pred, num_classes)
    print(f"Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"Avg Recall: {metrics['avg_recall']:.4f}")
    print(f"Avg F1: {metrics['avg_f1']:.4f}")
    
    # Confusion matrix
    confusion = compute_confusion_matrix(y_test, y_pred, num_classes)
    print("\nConfusion Matrix (first 5x5):")
    print(confusion[:5, :5])
    
    print("\n" + "=" * 60)
    print("✓ Complete pipeline test passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

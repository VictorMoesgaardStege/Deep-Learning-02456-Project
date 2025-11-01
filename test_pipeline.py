"""
Quick test of the complete example pipeline (without WandB)
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork
from utils import (
    one_hot_encode, 
    normalize_data, 
    compute_confusion_matrix,
    compute_metrics,
)
from wandb_logger import create_model_summary


def main():
    print("Quick Test of Neural Network Pipeline")
    print("=" * 60)
    
    # Set seed
    np.random.seed(42)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    num_classes = len(np.unique(y))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Normalize
    print("\n[2/5] Normalizing data...")
    X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)
    
    # Create model
    print("\n[3/5] Creating neural network...")
    layer_sizes = [X_train.shape[1], 64, 32, num_classes]
    activations = ['relu', 'relu', 'softmax']
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        learning_rate=0.05,
        l2_lambda=0.001,
        seed=42
    )
    
    print(create_model_summary(layer_sizes, activations))
    
    # Train
    print("\n[4/5] Training (20 epochs)...")
    history = model.train(
        X_train=X_train_norm,
        y_train=y_train_onehot,
        epochs=20,
        batch_size=32,
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

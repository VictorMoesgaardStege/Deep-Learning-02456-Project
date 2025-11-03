"""
Example: Training a Neural Network from Scratch on CIFAR-10 Dataset
This script demonstrates all features:
- Forward pass with matrix multiplications and activations
- Loss computation (Cross-Entropy) with L2 regularization
- Backward pass with manual gradient calculation
- Mini-batch gradient descent training
- Evaluation metrics (accuracy, loss curves, confusion matrix)
- WandB experiment tracking
"""

import numpy as np
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork
from helper_functions import (
    one_hot_encode, 
    normalize_data, 
    compute_confusion_matrix,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history
)
from wandb_logger import WandBLogger, create_model_summary
from cifar10_loader import load_cifar10_data, get_cifar10_class_names


def main():
    """Main function to demonstrate the neural network"""
    
    print("=" * 70)
    print("Neural Network from Scratch - Training on CIFAR-10")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ========================================
    # 1. Load and Prepare Data
    # ========================================
    print("\n[1/7] Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10 dataset (will use synthetic data if download fails)
    (X_train_full, y_train_full), (X_test, y_test) = load_cifar10_data(use_subset=False)
    
    num_classes = 10
    
    print(f"Dataset: {X_train_full.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train_full.shape[1]} (32x32x3 RGB images flattened)")
    print(f"Classes: {num_classes} (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)")
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ========================================
    # 2. Normalize Data
    # ========================================
    print("\n[2/7] Normalizing data...")
    
    X_train_norm, X_val_norm, mean, std = normalize_data(X_train, X_val)
    X_test_norm = (X_test - mean) / std
    
    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)
    
    print("Data normalization complete")
    
    # ========================================
    # 3. Define Model Architecture
    # ========================================
    print("\n[3/7] Creating neural network...")
    
    # Network architecture: Input -> Hidden1 -> Hidden2 -> Output
    # CIFAR-10 has 3072 input features (32x32x3)
    layer_sizes = [X_train.shape[1], 512, 256, num_classes]
    activations = ['relu', 'relu', 'softmax']
    
    # Hyperparameters
    config = {
        'layer_sizes': layer_sizes,
        'activations': activations,
        'learning_rate': 0.001,
        'l2_lambda': 0.0001,
        'epochs': 50,
        'batch_size': 128,
        'seed': 42,
        'dataset': 'CIFAR-10'
    }
    
    # Create model
    model = NeuralNetwork(
        layer_sizes=config['layer_sizes'],
        activations=config['activations'],
        learning_rate=config['learning_rate'],
        l2_lambda=config['l2_lambda'],
        seed=config['seed']
    )
    
    # Print model summary
    model_summary = create_model_summary(layer_sizes, activations)
    print(model_summary)
    
    # ========================================
    # 4. Initialize WandB Logger
    # ========================================
    print("\n[4/7] Initializing experiment tracking...")
    
    # Initialize WandB (set enabled=False to disable)
    wandb_logger = WandBLogger(
        project_name="neural-network-cifar10",
        config=config,
        enabled=True  # Set to False if you don't want to use WandB
    )
    
    wandb_logger.log_model_architecture(model_summary)
    
    # ========================================
    # 5. Train the Model
    # ========================================
    print("\n[5/7] Training neural network...")
    print("-" * 70)
    
    history = model.train(
        X_train=X_train_norm,
        y_train=y_train_onehot,
        X_val=X_val_norm,
        y_val=y_val_onehot,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        loss_type='cross_entropy',
        verbose=True
    )
    
    print("-" * 70)
    print("Training complete!")
    
    # Log training metrics to WandB
    for epoch in range(len(history['train_loss'])):
        wandb_logger.log_metrics({
            'epoch': epoch + 1,
            'train_loss': history['train_loss'][epoch],
            'train_accuracy': history['train_acc'][epoch],
            'val_loss': history['val_loss'][epoch],
            'val_accuracy': history['val_acc'][epoch]
        }, step=epoch)
    
    # ========================================
    # 6. Evaluate on Test Set
    # ========================================
    print("\n[6/7] Evaluating on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test_norm)
    test_accuracy = model.evaluate(X_test_norm, y_test_onehot)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # Compute detailed metrics
    metrics = compute_metrics(y_test, y_pred, num_classes)
    
    print(f"\nDetailed Metrics:")
    print(f"  Average Precision: {metrics['avg_precision']:.4f}")
    print(f"  Average Recall: {metrics['avg_recall']:.4f}")
    print(f"  Average F1 Score: {metrics['avg_f1']:.4f}")
    
    # Log test metrics to WandB
    wandb_logger.log_metrics({
        'test_accuracy': test_accuracy,
        'test_precision': metrics['avg_precision'],
        'test_recall': metrics['avg_recall'],
        'test_f1': metrics['avg_f1']
    })
    
    # Compute confusion matrix
    confusion = compute_confusion_matrix(y_test, y_pred, num_classes)
    
    print("\nConfusion Matrix:")
    print(confusion)
    
    # ========================================
    # 7. Visualize Results
    # ========================================
    print("\n[7/7] Generating visualizations...")
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')
    wandb_logger.log_image('training_history', 'training_history.png')
    
    # Plot confusion matrix
    class_names = get_cifar10_class_names()
    plot_confusion_matrix(confusion, class_names=class_names, save_path='confusion_matrix.png')
    wandb_logger.log_image('confusion_matrix', 'confusion_matrix.png')
    
    # Log confusion matrix to WandB
    wandb_logger.log_confusion_matrix(y_test, y_pred, class_names=class_names)
    
    print("\nVisualizations saved!")
    
    # ========================================
    # Finish
    # ========================================
    print("\n" + "=" * 70)
    print("Training and evaluation complete!")
    print("=" * 70)
    
    # Print final summary
    print(f"\nFinal Results:")
    print(f"  Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Finish WandB run
    wandb_logger.finish()
    
    print("\nAll done! Check the generated plots: training_history.png, confusion_matrix.png")


if __name__ == "__main__":
    main()

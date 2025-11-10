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
"""
Training a Neural Network on CIFAR-10 with WandB + Sweep Support
"""

import numpy as np
from sklearn.model_selection import train_test_split
import wandb

from neural_network import NeuralNetwork
from helper_functions import (
    one_hot_encode, 
    normalize_data, 
    compute_confusion_matrix,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history,
    create_model_summary
)
from wandb_logger import WandBLogger
from cifar10_loader import load_cifar10_data, get_cifar10_class_names


def main():
    print("=" * 70)
    print("Neural Network from Scratch - Training on CIFAR-10")
    print("=" * 70)

    np.random.seed(42)

    # 1. Load Data
    print("\n[1/7] Loading CIFAR-10 dataset...")
    (X_train_full, y_train_full), (X_test, y_test) = load_cifar10_data(use_subset=False)

    num_classes = 10

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    # 2. Normalize
    print("\n[2/7] Normalizing data...")
    X_train_norm, X_val_norm, mean, std = normalize_data(X_train, X_val)
    X_test_norm = (X_test - mean) / std

    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    # 3. Initialize WandB BEFORE referencing config
    print("\n[3/7] Initializing experiment tracking...")
    wandb_logger = WandBLogger(
        project_name="neural-network-cifar10",
        enabled=True
    )

    config = wandb.config  # <--- sweep parameters come from here

    # Build model architecture dynamically based on sweep
    input_dim = X_train.shape[1]
    layer_sizes = [input_dim] + config.hidden_layers + [num_classes]
    activations = config.activations

    print("\n[4/7] Creating neural network...")
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        learning_rate=config.learning_rate,
        l2_lambda=config.l2_lambda,
        optimizer=config.optimizer,
        weights_init=config.weights_init,
        seed=config.seed
    )

    model_summary = create_model_summary(layer_sizes, activations)
    print(model_summary)
    wandb_logger.log_model_architecture(model_summary)

    # 5. Train
    print("\n[5/7] Training neural network...")
    history = model.train(
        X_train=X_train_norm,
        y_train=y_train_onehot,
        X_val=X_val_norm,
        y_val=y_val_onehot,
        epochs=getattr(config, "epochs", 50),   # allow sweeps or default
        batch_size=config.batch_size,
        loss_type='cross_entropy',
        verbose=True
    )

    for epoch in range(len(history['train_loss'])):
        wandb_logger.log_metrics({
            'epoch': epoch + 1,
            'train_loss': history['train_loss'][epoch],
            'train_accuracy': history['train_acc'][epoch],
            'val_loss': history['val_loss'][epoch],
            'val_accuracy': history['val_acc'][epoch]
        }, step=epoch)

    # 6. Evaluate
    print("\n[6/7] Evaluating on test set...")
    y_pred = model.predict(X_test_norm)
    test_accuracy = model.evaluate(X_test_norm, y_test_onehot)

    metrics = compute_metrics(y_test, y_pred, num_classes)

    wandb_logger.log_metrics({
        'test_accuracy': test_accuracy,
        'test_precision': metrics['avg_precision'],
        'test_recall': metrics['avg_recall'],
        'test_f1': metrics['avg_f1']
    })

    confusion = compute_confusion_matrix(y_test, y_pred, num_classes)

    # 7. Visualizations
    print("\n[7/7] Generating visualizations...")
    plot_training_history(history, save_path='training_history.png')
    wandb_logger.log_image('training_history', 'training_history.png')

    class_names = get_cifar10_class_names()
    plot_confusion_matrix(confusion, class_names=class_names, save_path='confusion_matrix.png')
    wandb_logger.log_image('confusion_matrix', 'confusion_matrix.png')
    wandb_logger.log_confusion_matrix(y_test, y_pred, class_names=class_names)

    wandb_logger.finish()
    print("\nAll done!")


if __name__ == "__main__":
    main()

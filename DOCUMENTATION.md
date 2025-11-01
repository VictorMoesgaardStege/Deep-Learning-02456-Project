# Neural Network Implementation Guide

This document provides detailed information about the Neural Network implementation from scratch using NumPy.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Performance Tips](#performance-tips)

## Architecture Overview

### Components

The implementation consists of several key components:

#### 1. Activation Functions (`neural_network.py`)
- **ReLU**: f(x) = max(0, x)
- **Sigmoid**: f(x) = 1 / (1 + e^(-x))
- **Tanh**: f(x) = tanh(x)
- **Softmax**: f(x)_i = e^(x_i) / Σ e^(x_j)

Each activation function implements:
- `forward(x)`: Compute activation
- `backward(x)`: Compute gradient

#### 2. Layer Class
Represents a single fully-connected layer with:
- Weights (initialized using He initialization)
- Biases (initialized to zeros)
- Forward and backward propagation methods

#### 3. NeuralNetwork Class
Main class that orchestrates:
- Multiple layers
- Training with mini-batch SGD
- Loss computation
- Evaluation

## Mathematical Foundation

### Forward Propagation

For each layer l:
```
z[l] = W[l] · a[l-1] + b[l]
a[l] = activation(z[l])
```

Where:
- W[l]: Weight matrix for layer l
- b[l]: Bias vector for layer l
- a[l-1]: Activations from previous layer
- z[l]: Pre-activation values
- a[l]: Post-activation values

### Backward Propagation

Using the chain rule:

```
dL/dz[l] = dL/da[l] ⊙ activation'(z[l])
dL/dW[l] = (1/m) · a[l-1]^T · dL/dz[l]
dL/db[l] = (1/m) · sum(dL/dz[l])
dL/da[l-1] = dL/dz[l] · W[l]^T
```

Where:
- m: Batch size
- ⊙: Element-wise multiplication

### Loss Functions

#### Cross-Entropy Loss
```
L = -1/m Σ_i Σ_j y[i,j] · log(ŷ[i,j])
```

For softmax + cross-entropy, the gradient simplifies to:
```
dL/da[output] = ŷ - y
```

#### Mean Squared Error
```
L = 1/m Σ_i ||y[i] - ŷ[i]||²
```

#### L2 Regularization
```
L_reg = λ/(2m) Σ_l ||W[l]||²
```

Total loss:
```
L_total = L_data + L_reg
```

### Weight Update

Gradient descent update rule:
```
W[l] := W[l] - α · dL/dW[l]
b[l] := b[l] - α · dL/db[l]
```

Where α is the learning rate.

## API Reference

### NeuralNetwork Class

```python
NeuralNetwork(
    layer_sizes: List[int],
    activations: List[str] = None,
    learning_rate: float = 0.01,
    l2_lambda: float = 0.0,
    seed: Optional[int] = None
)
```

**Parameters:**
- `layer_sizes`: List of integers specifying the size of each layer [input_size, hidden1, ..., output_size]
- `activations`: List of activation function names for each layer. Options: 'relu', 'sigmoid', 'tanh', 'softmax'
- `learning_rate`: Learning rate for gradient descent (default: 0.01)
- `l2_lambda`: L2 regularization parameter (default: 0.0)
- `seed`: Random seed for reproducibility (default: None)

**Methods:**

#### train()
```python
train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    loss_type: str = 'cross_entropy',
    verbose: bool = True
) -> dict
```

Train the neural network.

**Parameters:**
- `X_train`: Training data, shape (n_samples, n_features)
- `y_train`: Training labels (one-hot encoded), shape (n_samples, n_classes)
- `X_val`: Validation data (optional)
- `y_val`: Validation labels (optional)
- `epochs`: Number of training epochs
- `batch_size`: Mini-batch size
- `loss_type`: 'cross_entropy' or 'mse'
- `verbose`: Print training progress

**Returns:**
- Dictionary with training history: `{'train_loss', 'train_acc', 'val_loss', 'val_acc'}`

#### predict()
```python
predict(X: np.ndarray) -> np.ndarray
```

Make predictions.

**Parameters:**
- `X`: Input data, shape (n_samples, n_features)

**Returns:**
- Predicted class indices, shape (n_samples,)

#### evaluate()
```python
evaluate(X: np.ndarray, y: np.ndarray) -> float
```

Evaluate accuracy.

**Parameters:**
- `X`: Input data
- `y`: True labels (one-hot encoded)

**Returns:**
- Accuracy score (float between 0 and 1)

### Utility Functions

#### one_hot_encode()
```python
one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray
```

Convert class indices to one-hot encoded vectors.

#### normalize_data()
```python
normalize_data(
    X_train: np.ndarray, 
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Normalize data using training set statistics.

**Returns:** `(X_train_norm, X_test_norm, mean, std)`

#### compute_confusion_matrix()
```python
compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    num_classes: int
) -> np.ndarray
```

Compute confusion matrix.

#### compute_metrics()
```python
compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    num_classes: int
) -> dict
```

Compute precision, recall, F1 score.

**Returns:** Dictionary with 'accuracy', 'precision', 'recall', 'f1', etc.

### CIFAR-10 Loader

#### load_cifar10_data()
```python
from cifar10_loader import load_cifar10_data, get_cifar10_class_names

load_cifar10_data(use_subset: bool = False) -> Tuple
```

Load CIFAR-10 dataset with automatic fallback to synthetic data if download fails.

**Parameters:**
- `use_subset`: If True, uses smaller subset for quick testing (5000 train, 1000 test)

**Returns:** `((X_train, y_train), (X_test, y_test))` with images flattened to 3072 features

## Usage Examples

### Example 1: Training on CIFAR-10

```python
import numpy as np
from neural_network import NeuralNetwork
from utils import one_hot_encode, normalize_data
from cifar10_loader import load_cifar10_data, get_cifar10_class_names

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = load_cifar10_data()

# Normalize
X_train_norm, X_test_norm, _, _ = normalize_data(X_train, X_test)
y_train_onehot = one_hot_encode(y_train, num_classes=10)

# Create model for CIFAR-10 (3072 input features)
model = NeuralNetwork(
    layer_sizes=[3072, 512, 256, 10],
    activations=['relu', 'relu', 'softmax'],
    learning_rate=0.001,
    l2_lambda=0.0001
)

# Train
history = model.train(
    X_train=X_train_norm,
    y_train=y_train_onehot,
    epochs=50,
    batch_size=128
)

# Evaluate
predictions = model.predict(X_test_norm)
class_names = get_cifar10_class_names()
```

### Example 2: Basic Classification

```python
import numpy as np
from neural_network import NeuralNetwork
from utils import one_hot_encode, normalize_data

# Prepare data
X_train = np.random.randn(1000, 20)  # 1000 samples, 20 features
y_train = np.random.randint(0, 5, 1000)  # 5 classes

# Preprocess
X_train_norm, X_test_norm, _, _ = normalize_data(X_train, X_test)
y_train_onehot = one_hot_encode(y_train, num_classes=5)

# Create model
model = NeuralNetwork(
    layer_sizes=[20, 64, 32, 5],
    activations=['relu', 'relu', 'softmax'],
    learning_rate=0.01,
    l2_lambda=0.001
)

# Train
history = model.train(
    X_train=X_train_norm,
    y_train=y_train_onehot,
    epochs=100,
    batch_size=32
)

# Predict
predictions = model.predict(X_test_norm)
```

### Example 3: With Validation Set

```python
# Train with validation
history = model.train(
    X_train=X_train_norm,
    y_train=y_train_onehot,
    X_val=X_val_norm,
    y_val=y_val_onehot,
    epochs=100,
    batch_size=32,
    verbose=True
)

# Access training history
print(f"Final train loss: {history['train_loss'][-1]}")
print(f"Final val accuracy: {history['val_acc'][-1]}")
```

### Example 3: Different Activation Functions

```python
# Network with different activations
model = NeuralNetwork(
    layer_sizes=[20, 64, 32, 16, 5],
    activations=['tanh', 'relu', 'sigmoid', 'softmax'],
    learning_rate=0.001,
    l2_lambda=0.01
)
```

### Example 4: Regression Task

```python
# For regression, use linear output (no activation or sigmoid)
# and MSE loss
model = NeuralNetwork(
    layer_sizes=[10, 32, 16, 1],
    activations=['relu', 'relu', 'sigmoid'],
    learning_rate=0.01
)

history = model.train(
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    loss_type='mse'
)
```

### Example 5: Complete Pipeline with Visualization

```python
from utils import (
    plot_training_history, 
    plot_confusion_matrix,
    compute_confusion_matrix,
    compute_metrics
)

# Train model
history = model.train(X_train_norm, y_train_onehot, epochs=100)

# Evaluate
y_pred = model.predict(X_test_norm)
accuracy = model.evaluate(X_test_norm, y_test_onehot)

# Compute metrics
metrics = compute_metrics(y_test, y_pred, num_classes)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['avg_f1']:.4f}")

# Visualize
plot_training_history(history, save_path='history.png')

confusion = compute_confusion_matrix(y_test, y_pred, num_classes)
plot_confusion_matrix(confusion, save_path='confusion.png')
```

## Performance Tips

### 1. Learning Rate Selection
- Start with 0.01 and adjust based on loss curves
- If loss oscillates: decrease learning rate
- If loss decreases too slowly: increase learning rate

### 2. Architecture Design
- Start simple (1-2 hidden layers)
- Use ReLU for hidden layers (faster than sigmoid/tanh)
- Output layer: softmax for classification, sigmoid for binary, linear for regression

### 3. Regularization
- Use L2 regularization (λ = 0.001 to 0.01) to prevent overfitting
- Monitor validation loss to detect overfitting

### 4. Batch Size
- Larger batches (128-256): faster, more stable gradients
- Smaller batches (16-32): better generalization, more updates per epoch
- Trade-off between speed and accuracy

### 5. Data Preprocessing
- Always normalize features (zero mean, unit variance)
- Shuffle training data each epoch
- Use validation set for hyperparameter tuning

### 6. Initialization
- Default He initialization works well for ReLU
- For sigmoid/tanh, consider Xavier initialization

### 7. Monitoring Training
- Track both training and validation metrics
- Stop if validation loss stops improving (early stopping)
- Loss should generally decrease; if not, check learning rate

## Common Issues and Solutions

### Issue: Loss is NaN
- **Cause**: Learning rate too high, numerical overflow
- **Solution**: Reduce learning rate, check for extreme input values

### Issue: Loss not decreasing
- **Cause**: Learning rate too low, poor initialization
- **Solution**: Increase learning rate, check data preprocessing

### Issue: Training accuracy high, test accuracy low
- **Cause**: Overfitting
- **Solution**: Increase L2 regularization, reduce model complexity, get more data

### Issue: Both train and test accuracy low
- **Cause**: Underfitting
- **Solution**: Increase model capacity, train longer, reduce regularization

### Issue: Slow training
- **Cause**: Small batch size, large model
- **Solution**: Increase batch size, reduce model size

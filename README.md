# Neural Network from Scratch with NumPy

A complete implementation of a feedforward neural network from scratch using only NumPy, featuring training, optimization, and experiment tracking with Weights & Biases (WandB).

## 🎯 Objective

Design and train a feedforward neural network (FFN) from scratch using only NumPy, without deep learning libraries such as TensorFlow or PyTorch.

## ✨ Features

### Core Implementation
- **Forward Pass**: Matrix multiplications with multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
- **Loss Computation**: Mean Squared Error (MSE) and Cross-Entropy loss with L2 regularization
- **Backward Pass**: Manual derivative calculation and weight updates using backpropagation
- **Training Loop**: Mini-batch gradient descent with configurable batch size
- **Evaluation**: Compute accuracy, loss curves, confusion matrix, precision, recall, and F1 score

### Additional Features
- Multiple activation functions support
- L2 regularization to prevent overfitting
- Data normalization utilities
- One-hot encoding for classification
- Comprehensive visualization tools
- WandB integration for experiment tracking

## 📁 Project Structure

```
.
├── neural_network.py    # Core neural network implementation
├── utils.py            # Evaluation and visualization utilities
├── wandb_logger.py     # WandB integration for experiment tracking
├── example.py          # Complete training example
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VictorMoesgaardStege/Deep-Learning-02456-Project.git
cd Deep-Learning-02456-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Example

Run the complete training example on the digits dataset:

```bash
python example.py
```

This will:
- Load and preprocess the digits dataset
- Train a neural network with 2 hidden layers
- Evaluate on a test set
- Generate loss/accuracy curves and confusion matrix
- Log everything to WandB (if enabled)

## 📖 Usage

### Basic Usage

```python
from neural_network import NeuralNetwork
from utils import one_hot_encode, normalize_data

# Prepare data
X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
y_train_onehot = one_hot_encode(y_train, num_classes)

# Create model
model = NeuralNetwork(
    layer_sizes=[64, 128, 64, 10],  # Input -> Hidden1 -> Hidden2 -> Output
    activations=['relu', 'relu', 'softmax'],
    learning_rate=0.01,
    l2_lambda=0.001,
    seed=42
)

# Train model
history = model.train(
    X_train=X_train_norm,
    y_train=y_train_onehot,
    X_val=X_val_norm,
    y_val=y_val_onehot,
    epochs=100,
    batch_size=32,
    loss_type='cross_entropy',
    verbose=True
)

# Make predictions
predictions = model.predict(X_test_norm)

# Evaluate
accuracy = model.evaluate(X_test_norm, y_test_onehot)
```

### Activation Functions

Available activation functions:
- `'relu'`: Rectified Linear Unit
- `'sigmoid'`: Sigmoid function
- `'tanh'`: Hyperbolic tangent
- `'softmax'`: Softmax (typically for output layer in classification)

### Loss Functions

- `'cross_entropy'`: Cross-entropy loss for classification
- `'mse'`: Mean Squared Error for regression

### Visualization

```python
from utils import plot_training_history, plot_confusion_matrix, compute_confusion_matrix

# Plot training history
plot_training_history(history, save_path='training_history.png')

# Compute and plot confusion matrix
confusion = compute_confusion_matrix(y_true, y_pred, num_classes)
plot_confusion_matrix(confusion, class_names=['0', '1', '2', ...], save_path='confusion_matrix.png')
```

### WandB Integration

```python
from wandb_logger import WandBLogger

# Initialize logger
wandb_logger = WandBLogger(
    project_name="my-project",
    config={'learning_rate': 0.01, 'epochs': 100},
    enabled=True
)

# Log metrics
wandb_logger.log_metrics({'train_loss': loss, 'train_acc': acc}, step=epoch)

# Log confusion matrix
wandb_logger.log_confusion_matrix(y_true, y_pred, class_names)

# Finish run
wandb_logger.finish()
```

## 🧮 Mathematical Details

### Forward Pass

For each layer `l`:
```
z[l] = W[l] · a[l-1] + b[l]
a[l] = activation(z[l])
```

### Backward Pass

Gradients computed using chain rule:
```
dL/dW[l] = (1/m) · a[l-1]ᵀ · dz[l]
dL/db[l] = (1/m) · sum(dz[l])
dL/da[l-1] = dz[l] · W[l]ᵀ
```

### Loss Functions

**Cross-Entropy Loss:**
```
L = -1/m Σ Σ y[i,j] · log(ŷ[i,j])
```

**MSE Loss:**
```
L = 1/m Σ ||y[i] - ŷ[i]||²
```

**L2 Regularization:**
```
L_reg = λ/(2m) Σ ||W[l]||²
```

## 📊 Example Results

The example script trains on the digits dataset (8x8 images of handwritten digits) and typically achieves:
- Training Accuracy: ~99%
- Test Accuracy: ~97%

Results include:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix showing per-class performance
- Precision, recall, and F1 scores

## 🛠️ Implementation Details

### Neural Network Architecture

The `NeuralNetwork` class supports:
- Arbitrary number of layers
- Different activation functions per layer
- L2 regularization
- Mini-batch gradient descent
- Configurable learning rate

### Key Components

1. **Layer Class**: Encapsulates weights, biases, and activations for a single layer
2. **Activation Functions**: Modular implementation with forward and backward methods
3. **Training Loop**: Mini-batch SGD with shuffling and validation
4. **Evaluation**: Accuracy, confusion matrix, and detailed metrics

## 📝 Requirements

- Python 3.7+
- NumPy >= 1.21.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- wandb >= 0.13.0 (optional, for experiment tracking)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

This implementation is designed for educational purposes to understand the fundamentals of neural networks and backpropagation.

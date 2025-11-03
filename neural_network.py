"""
Neural Network Implementation from Scratch using NumPy
This module implements a feedforward neural network with:
- Forward pass with matrix multiplications and activation functions
- Loss computation (MSE and Cross-Entropy) with L2 regularization
- Backward pass with manual gradient calculation
- Mini-batch gradient descent training
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


class ActivationFunction:
    """Base class for activation functions with forward and backward methods"""
    # If activation is not specified, raise error
    
    @staticmethod
    def forward(x):
        raise NotImplementedError
    
    @staticmethod
    def backward(x):
        raise NotImplementedError


class ReLU(ActivationFunction):
    """ReLU activation function""" 

    #ReuLU activation function for forward
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    #ReuLU activation function for backward pass (derivative of relu)
    @staticmethod
    def backward(x):
        return (x > 0).astype(float) # Derivative is 1 for x>0, else 0 (this is a logical test)


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    
    @staticmethod
    def forward(x):
        # Clip to prevent overflow (every value is squashed between -500 and 500)
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def backward(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    """Tanh activation function"""
    
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(x):
        return 1 - np.tanh(x) ** 2


class Softmax(ActivationFunction):
    """Softmax activation function for multi-class classification"""
    
    @staticmethod
    def forward(x):
        # Subtract max for numerical stability - it makes no difference to the result if we subtract a constant from all inputs. Avoids large exponents problems.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        # Actual gradient calculation is done in the NeuralNetwork.backward method
        # and this is a placeholder that multiplies gradient by 1. 
        return np.ones_like(x)


class Layer:
    """Single layer in the neural network"""
    
    def __init__(self, input_size: int, output_size: int, activation: ActivationFunction):
        """
        Initialize layer with random weights and zero biases
        
        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            activation: Activation function to use
        """
        # He normal initialization for weights: W ~ N(0, sqrt(2/input size)) 
        # This initialization works well with ReLU activations (https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # Cache for backward pass (memory trick for storing intermediate calculation values)
        self.input = None
        self.z = None  # Pre-activation
        self.a = None  # Post-activation
        
        # Gradients - the parameters to be updated during training
        self.dweights = None
        self.dbiases = None
    
    def forward(self, x):
        """
        Forward pass through the layer
        
        Args:
            x: Input data of shape (batch_size, input_size)
            
        Returns:
            Activated output of shape (batch_size, output_size)
        """
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, da):
        """
        Backward pass through the layer
        
        Args:
            da: Gradient of loss with respect to layer output (meaning the activated output from given layer)
            
        Returns:
            Gradient of loss with respect to layer input
        """
        # Gradient through activation 
        dz = da * self.activation.backward(self.z) # actually dL/dz = dL/da * da/dz
        
        # Gradient with respect to weights and biases
        batch_size = self.input.shape[0]
        self.dweights = np.dot(self.input.T, dz) / batch_size
        self.dbiases = np.sum(dz, axis=0, keepdims=True) / batch_size # average of dz (batch_size,output_size) over batch
        
        # Gradient with respect to input
        dx = np.dot(dz, self.weights.T) # dL/dx = dL/dz * dz/dx (because dz/dx = W.T)
        
        return dx


class NeuralNetwork:
    """
    Feedforward Neural Network implemented from scratch using NumPy
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 activations: List[str] = None,
                 learning_rate: float = 0.01,
                 l2_lambda: float = 0.0,
                 seed: Optional[int] = None):
        """
        Initialize the neural network
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation function names for each layer
            learning_rate: Learning rate for gradient descent
            l2_lambda: L2 regularization parameter
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.layers = []
        
        # # Default activations: ReLU for hidden layers, Softmax for output
        # if activations is None:
        #     activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
        
        # Map activation names to classes
        activation_map = {
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'softmax': Softmax
        }
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            activation_name = activations[i].lower()
            activation = activation_map.get(activation_name, ReLU)
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)
        
        # Training history
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
    
    def forward(self, x):
        """
        Forward pass through the entire network
        
        Args:
            x: Input data of shape (batch_size, input_size)
            
        Returns:
            Network output of shape (batch_size, output_size)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_pred, y_true):
        """
        Backward pass through the entire network
        
        Args:
            y_pred: Predicted values of shape (batch_size, output_size)
            y_true: True values of shape (batch_size, output_size)
        """
        # Gradient of loss with respect to output
        # For softmax + cross-entropy: gradient is simply (y_pred - y_true)
        da = y_pred - y_true
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            da = layer.backward(da)
    
    def compute_loss(self, y_pred, y_true, loss_type: str = 'cross_entropy'):
        """
        Compute loss with L2 regularization
        
        Args:
            y_pred: Predicted values
            y_true: True values
            loss_type: Type of loss ('mse' or 'cross_entropy') - maybe implement mse later
            
        Returns:
            Loss value
        """
        # Constants for numerical stability
        EPSILON_CLIP = 1e-10  # Small value to prevent log(0) and division by zero
        MAX_CLIP = 1 - 1e-10  # Maximum value for clipping probabilities

        batch_size = y_true.shape[0]
        
        if loss_type == 'mse':
            # Mean Squared Error
            data_loss = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
        elif loss_type == 'cross_entropy':
            # Cross-Entropy Loss
            # Clip predictions to prevent log(0)
            y_pred_clipped = np.clip(y_pred, EPSILON_CLIP, MAX_CLIP)
            data_loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1)) # negative log likelihood
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # L2 Regularization
        l2_loss = 0
        if self.l2_lambda > 0:
            for layer in self.layers:
                l2_loss += np.sum(layer.weights ** 2)
            l2_loss *= (self.l2_lambda / (2 * batch_size))
        
        return data_loss + l2_loss
    
    def update_weights(self):
        """Update weights and biases using gradients"""
        for layer in self.layers:
            # Add L2 regularization gradient to weight gradients
            if self.l2_lambda > 0:
                layer.dweights += self.l2_lambda * layer.weights # 1/m is already included in layer.dweights
            
            # Update weights and biases
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases
    
    def train(self, 
              X_train,
              y_train,
              X_val = None,
              y_val = None,
              epochs = 100,
              batch_size = 32,
              loss_type: str = 'cross_entropy',
              verbose: bool = True):
        """
        Train the neural network using mini-batch gradient descent
        
        Args:
            X_train: Training data of shape (n_samples, n_features)
            y_train: Training labels (one-hot encoded for classification)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            loss_type: Type of loss ('mse' or 'cross_entropy')
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size) # Ensure at least one batch exists (// is floor division)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples) # random permutation of indices
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_pred, y_batch, loss_type)
                epoch_loss += batch_loss
                
                # Backward pass
                self.backward(y_pred, y_batch) # compute gradients and store in layers
                
                # Update weights
                self.update_weights()
            
            # Average loss. Epoch loss per batch
            avg_loss = epoch_loss / n_batches
            self.train_loss_history.append(avg_loss)
            
            # Compute training accuracy
            train_acc = self.evaluate(X_train, y_train)
            self.train_acc_history.append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val, loss_type)
                val_acc = self.evaluate(X_val, y_val)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")
        
        return {
            'train_loss': self.train_loss_history,
            'train_acc': self.train_acc_history,
            'val_loss': self.val_loss_history,
            'val_acc': self.val_acc_history
        }
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted class indices
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate accuracy
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

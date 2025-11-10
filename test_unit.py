"""
Unit tests for the Neural Network implementation
Tests individual components: activations, layers, loss functions, etc.
"""

import numpy as np
import sys

def test_activation_functions():
    """Test activation functions forward and backward passes"""
    from neural_network import ReLU, Sigmoid, Tanh, Softmax
    
    print("Testing Activation Functions...")
    
    # Test ReLU
    x = np.array([[-1, 0, 1, 2]])
    relu_output = ReLU.forward(x)
    assert np.array_equal(relu_output, np.array([[0, 0, 1, 2]])), "ReLU forward failed"
    
    relu_grad = ReLU.backward(x)
    assert np.array_equal(relu_grad, np.array([[0, 0, 1, 1]])), "ReLU backward failed"
    
    # Test Sigmoid
    x = np.array([[0]])
    sigmoid_output = Sigmoid.forward(x)
    assert np.isclose(sigmoid_output[0, 0], 0.5), "Sigmoid forward failed"
    
    # Test Softmax
    x = np.array([[1, 2, 3]])
    softmax_output = Softmax.forward(x)
    assert np.isclose(np.sum(softmax_output), 1.0), "Softmax sum should be 1"
    
    print("  ✓ All activation tests passed")


def test_layer():
    """Test Layer forward and backward passes"""
    from neural_network import Layer, ReLU
    
    print("Testing Layer...")
    
    np.random.seed(42)
    layer = Layer(input_size=5, output_size=3, activation=ReLU, weights_init="he")
    
    # Test shapes
    assert layer.weights.shape == (5, 3), "Weight shape incorrect"
    assert layer.biases.shape == (1, 3), "Bias shape incorrect"
    
    # Test forward
    x = np.random.randn(10, 5)  # batch of 10
    output = layer.forward(x)
    assert output.shape == (10, 3), "Layer output shape incorrect"
    
    # Test backward
    da = np.random.randn(10, 3)
    dx = layer.backward(da)
    assert dx.shape == (10, 5), "Layer backward output shape incorrect"
    assert layer.dweights.shape == (5, 3), "Weight gradient shape incorrect"
    assert layer.dbiases.shape == (1, 3), "Bias gradient shape incorrect"
    
    print("  ✓ All layer tests passed")


def test_loss_functions():
    """Test loss computation"""
    from neural_network import NeuralNetwork
    
    print("Testing Loss Functions...")
    
    np.random.seed(42)
    model = NeuralNetwork(layer_sizes=[5, 3], activations=['softmax'], l2_lambda=0.0)
    
    # Test MSE
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    
    mse_loss = model.compute_loss(y_pred, y_true, 'mse')
    assert mse_loss > 0, "MSE loss should be positive"
    
    # Test Cross-Entropy
    ce_loss = model.compute_loss(y_pred, y_true, 'cross_entropy')
    assert ce_loss > 0, "Cross-entropy loss should be positive"
    
    # Test with L2 regularization
    model.l2_lambda = 0.1
    ce_loss_reg = model.compute_loss(y_pred, y_true, 'cross_entropy')
    assert ce_loss_reg > ce_loss, "Loss with L2 should be higher"
    
    print("  ✓ All loss function tests passed")


def test_gradient_numerical():
    """Test gradients using numerical approximation"""
    from neural_network import NeuralNetwork
    
    print("Testing Gradient Computation (numerical check)...")
    
    np.random.seed(42)
    
    # Small model for gradient checking
    model = NeuralNetwork(
        layer_sizes=[3, 4, 2],
        activations=['relu', 'softmax'],
        l2_lambda=0.0,
        seed=42
    )
    
    # Small dataset
    X = np.random.randn(5, 3)
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    
    # Compute gradients
    y_pred = model.forward(X)
    model.backward(y_pred, y)
    
    # Check first layer weights gradient numerically
    layer = model.layers[0]
    analytical_grad = layer.dweights.copy()
    
    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.weights)
    
    # Check a few weights (not all for speed)
    for i in range(min(2, layer.weights.shape[0])):
        for j in range(min(2, layer.weights.shape[1])):
            # Perturb weight
            original = layer.weights[i, j]
            
            layer.weights[i, j] = original + epsilon
            y_pred_plus = model.forward(X)
            loss_plus = model.compute_loss(y_pred_plus, y, 'cross_entropy')
            
            layer.weights[i, j] = original - epsilon
            y_pred_minus = model.forward(X)
            loss_minus = model.compute_loss(y_pred_minus, y, 'cross_entropy')
            
            layer.weights[i, j] = original
            
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Check if analytical and numerical gradients are close
    for i in range(min(2, layer.weights.shape[0])):
        for j in range(min(2, layer.weights.shape[1])):
            diff = abs(analytical_grad[i, j] - numerical_grad[i, j])
            relative_error = diff / (abs(analytical_grad[i, j]) + abs(numerical_grad[i, j]) + 1e-8)
            assert relative_error < 1e-5, f"Gradient check failed: {relative_error}"
    
    print("  ✓ Gradient numerical check passed")


def test_optimizers():
    """Test that different optimizers update weights differently and reduce loss."""
    from neural_network import NeuralNetwork
    from helper_functions import one_hot_encode
    
    print("Testing Optimizers (SGD, Momentum, Adam)...")
    
    np.random.seed(42)
    
    # Simple dataset
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 3, 50)
    y_onehot = one_hot_encode(y, 3)
    
    # Create 3 identical models with different optimizers
    model_sgd = NeuralNetwork(layer_sizes=[10, 8, 3],
                              activations=['relu', 'softmax'],
                              learning_rate=0.1,
                              optimizer="sgd",
                              seed=42)
    
    model_momentum = NeuralNetwork(layer_sizes=[10, 8, 3],
                                   activations=['relu', 'softmax'],
                                   learning_rate=0.1,
                                   optimizer="momentum",
                                   seed=42)
    
    model_adam = NeuralNetwork(layer_sizes=[10, 8, 3],
                               activations=['relu', 'softmax'],
                               learning_rate=0.1,
                               optimizer="adam",
                               seed=42)
    
    # Train briefly (just enough to show change)
    hist_sgd = model_sgd.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)
    hist_momentum = model_momentum.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)
    hist_adam = model_adam.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)
    
    # Loss should decrease
    assert hist_sgd['train_loss'][-1] < hist_sgd['train_loss'][0], "SGD should reduce loss"
    assert hist_momentum['train_loss'][-1] < hist_momentum['train_loss'][0], "Momentum should reduce loss"
    assert hist_adam['train_loss'][-1] < hist_adam['train_loss'][0], "Adam should reduce loss"
    
    # Weight updates should differ between optimizers
    w_sgd = model_sgd.layers[0].weights
    w_momentum = model_momentum.layers[0].weights
    w_adam = model_adam.layers[0].weights
    
    assert not np.allclose(w_sgd, w_momentum), "SGD and Momentum should produce different updates"
    assert not np.allclose(w_sgd, w_adam), "SGD and Adam should produce different updates"
    assert not np.allclose(w_momentum, w_adam), "Momentum and Adam should produce different updates"
    
    print("  ✓ Optimizer tests passed")


def test_weight_initialization():
    """Test whether He and Xavier initializations produce correct weight scale."""
    from neural_network import Layer, ReLU
    
    print("Testing Weight Initialization...")
    np.random.seed(42)

    fan_in = 20
    fan_out = 10

    # Test He initialization
    layer_he = Layer(fan_in, fan_out, activation=ReLU, weights_init="he")
    he_std = np.std(layer_he.weights)
    expected_he_std = np.sqrt(2.0 / fan_in)
    assert np.isclose(he_std, expected_he_std, rtol=0.25), (
        f"He init std incorrect: got {he_std}, expected approx {expected_he_std}"
    )

    # Test Xavier initialization
    layer_xavier = Layer(fan_in, fan_out, activation=ReLU, weights_init="xavier")
    xav_std = np.std(layer_xavier.weights)
    expected_xav_std = np.sqrt(2.0 / (fan_in + fan_out))
    assert np.isclose(xav_std, expected_xav_std, rtol=0.25), (
        f"Xavier init std incorrect: got {xav_std}, expected approx {expected_xav_std}"
    )

    print("  ✓ Weight initialization tests passed")




def test_training():
    """Test training loop"""
    from neural_network import NeuralNetwork
    from helper_functions import one_hot_encode
    
    print("Testing Training Loop...")
    
    np.random.seed(42)
    
    # Create simple dataset
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    y_train_onehot = one_hot_encode(y_train, 3)
    
    # Create model
    model = NeuralNetwork(
        layer_sizes=[10, 8, 3],
        activations=['relu', 'softmax'],
        learning_rate=0.1,
        seed=42
    )
    
    # Train
    history = model.train(
        X_train=X_train,
        y_train=y_train_onehot,
        epochs=20,
        batch_size=32,
        verbose=False
    )
    
    # Check that loss decreased
    assert history['train_loss'][-1] < history['train_loss'][0], "Loss should decrease during training"
    
    # Check that accuracy improved
    assert history['train_acc'][-1] > history['train_acc'][0], "Accuracy should improve during training"
    
    print("  ✓ Training loop test passed")


def test_utils():
    """Test utility functions"""
    from helper_functions import (
        one_hot_encode, 
        normalize_data, 
        compute_confusion_matrix,
        compute_metrics
    )
    
    print("Testing Utility Functions...")
    
    # Test one-hot encoding
    y = np.array([0, 1, 2, 0, 1])
    y_onehot = one_hot_encode(y, 3)
    assert y_onehot.shape == (5, 3), "One-hot shape incorrect"
    assert np.all(y_onehot.sum(axis=1) == 1), "One-hot should sum to 1"
    
    # Test normalization
    X_train = np.random.randn(100, 10) * 5 + 10
    X_test = np.random.randn(50, 10) * 5 + 10
    
    X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
    assert np.abs(np.mean(X_train_norm)) < 0.1, "Mean should be close to 0"
    assert np.abs(np.std(X_train_norm) - 1.0) < 0.1, "Std should be close to 1"
    
    # Test confusion matrix
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2])
    confusion = compute_confusion_matrix(y_true, y_pred, 3)
    assert confusion.shape == (3, 3), "Confusion matrix shape incorrect"
    assert confusion[0, 0] == 1, "Confusion matrix value incorrect"
    assert confusion[0, 1] == 1, "Confusion matrix value incorrect"
    
    # Test metrics
    metrics = compute_metrics(y_true, y_pred, 3)
    assert 'accuracy' in metrics, "Metrics should include accuracy"
    assert 'precision' in metrics, "Metrics should include precision"
    assert 'recall' in metrics, "Metrics should include recall"
    
    print("  ✓ All utility tests passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Unit Tests for Neural Network Implementation")
    print("=" * 60)
    print()
    
    try:
        test_activation_functions()
        test_layer()
        test_loss_functions()
        test_gradient_numerical()
        test_training()
        test_optimizers()
        test_weight_initialization()
        test_utils()
        
        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ ERROR: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

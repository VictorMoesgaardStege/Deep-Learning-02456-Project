"""
Unit tests for the Neural Network implementation
Covers activations, layers, losses, gradients, optimizers, and training loop.
"""

import numpy as np
import sys


# ============================================================
# ACTIVATION FUNCTION TESTS
# ============================================================
def test_activation_functions():
    from neural_network import ReLU, Sigmoid, Tanh, Identity, Softmax

    print("Testing Activation Functions...")

    # ---------- ReLU ----------
    x = np.array([[-1.0, 0.0, 1.0, 2.0]])
    relu_out = ReLU.forward(x)
    relu_expected = np.array([[0.0, 0.0, 1.0, 2.0]])
    assert np.allclose(relu_out, relu_expected), "ReLU forward incorrect"

    relu_grad = ReLU.backward(x, relu_out, np.ones_like(x))
    relu_grad_expected = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert np.allclose(relu_grad, relu_grad_expected), "ReLU backward incorrect"

    # ---------- Sigmoid ----------
    sig_out = Sigmoid.forward(np.array([[0.0]]))
    assert np.isclose(sig_out[0, 0], 0.5), "Sigmoid forward incorrect"

    # ---------- Tanh ----------
    tanh_out = Tanh.forward(np.array([[0.0]]))
    assert np.isclose(tanh_out[0, 0], 0.0), "Tanh forward incorrect"

    # ---------- Identity ----------
    id_in = np.array([[5.0, -3.0]])
    id_out = Identity.forward(id_in)
    assert np.allclose(id_out, id_in), "Identity forward incorrect"

    id_grad = Identity.backward(id_in, id_out, np.ones_like(id_in))
    assert np.allclose(id_grad, np.ones_like(id_in)), "Identity backward incorrect"

    # ---------- Softmax ----------
    sm = Softmax.forward(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(np.sum(sm), 1.0), "Softmax output does not sum to 1"

    print("  ✓ Activation function tests passed")


# ============================================================
# LAYER TESTS
# ============================================================
def test_layer():
    from neural_network import Layer, ReLU

    print("Testing Layer...")

    np.random.seed(42)
    layer = Layer(input_size=5, output_size=3, activation=ReLU, weights_init="he")

    x = np.random.randn(10, 5)
    out = layer.forward(x)
    assert out.shape == (10, 3), "Layer forward wrong output shape"

    grad_out = np.random.randn(10, 3)
    dx = layer.backward(grad_out)

    assert dx.shape == (10, 5), "Layer backward wrong dx shape"
    assert layer.dweights.shape == (5, 3), "Incorrect weight gradient shape"
    assert layer.dbiases.shape == (1, 3), "Incorrect bias gradient shape"

    print("  ✓ Layer tests passed")


# ============================================================
# LOSS FUNCTION TESTS
# ============================================================
def test_loss_functions():
    from neural_network import NeuralNetwork

    print("Testing Loss Functions...")

    np.random.seed(42)
    model = NeuralNetwork(layer_sizes=[5, 3], activations=['softmax'])

    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.05, 0.05],
                       [0.1, 0.8, 0.1]])

    # ----- MSE -----
    model.loss_type = "mse"
    mse_loss = model.compute_loss(y_pred, y_true)
    assert mse_loss > 0, "MSE loss must be positive"

    # ----- Cross-Entropy -----
    model.loss_type = "cross_entropy"
    ce_loss = model.compute_loss(y_pred, y_true)
    assert ce_loss > 0, "CE loss must be positive"

    # ----- L2 regularization -----
    model.l2_lambda = 0.1
    ce_l2 = model.compute_loss(y_pred, y_true)
    assert ce_l2 > ce_loss, "L2 should increase total loss"

    print("  ✓ Loss function tests passed")


# ============================================================
# GRADIENT CHECKING
# ============================================================
def test_gradient_numerical():
    from neural_network import NeuralNetwork

    print("Testing Gradient Checking...")

    np.random.seed(42)
    model = NeuralNetwork(
        layer_sizes=[3, 4, 2],
        activations=['relu', 'softmax'],
        seed=42
    )

    X = np.random.randn(5, 3)
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])

    # Forward + backprop
    y_pred = model.forward(X)
    model.loss_type = "cross_entropy"
    model.backward(y_pred, y)

    layer = model.layers[0]
    analytical = layer.dweights.copy()

    eps = 1e-5
    numerical = np.zeros_like(layer.weights)

    for i in range(2):
        for j in range(2):
            w_orig = layer.weights[i, j]

            layer.weights[i, j] = w_orig + eps
            plus = model.compute_loss(model.forward(X), y)

            layer.weights[i, j] = w_orig - eps
            minus = model.compute_loss(model.forward(X), y)

            layer.weights[i, j] = w_orig

            numerical[i, j] = (plus - minus) / (2 * eps)

            rel_err = np.abs(analytical[i, j] - numerical[i, j]) / (
                np.abs(analytical[i, j]) + np.abs(numerical[i, j]) + 1e-8
            )
            assert rel_err < 1e-4, f"Gradient mismatch: rel_err={rel_err}"

    print("  ✓ Gradient numerical check passed")


# ============================================================
# OPTIMIZER TESTS
# ============================================================
def test_optimizers():
    from neural_network import NeuralNetwork
    from helper_functions import one_hot_encode

    print("Testing Optimizers...")

    np.random.seed(42)

    X = np.random.randn(50, 10)
    y = np.random.randint(0, 3, 50)
    y_onehot = one_hot_encode(y, 3)

    model_sgd = NeuralNetwork([10, 8, 3], ['relu', 'softmax'], learning_rate=0.1, optimizer="sgd", seed=42)
    model_mom = NeuralNetwork([10, 8, 3], ['relu', 'softmax'], learning_rate=0.1, optimizer="momentum", seed=42)
    model_nest = NeuralNetwork([10, 8, 3], ['relu', 'softmax'], learning_rate=0.1, optimizer="nesterov", seed=42)
    model_adam = NeuralNetwork([10, 8, 3], ['relu', 'softmax'], learning_rate=0.1, optimizer="adam", seed=42)

    h_sgd = model_sgd.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)
    h_mom = model_mom.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)
    h_nest = model_nest.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)
    h_adam = model_adam.train(X, y_onehot, epochs=5, batch_size=16, verbose=False)

    # Loss should drop
    for h in [h_sgd, h_mom, h_nest, h_adam]:
        assert h["train_loss"][-1] < h["train_loss"][0], "Optimizer did not reduce loss"

    # Optimizers should produce different weights
    assert not np.allclose(model_sgd.layers[0].weights, model_mom.layers[0].weights)
    assert not np.allclose(model_sgd.layers[0].weights, model_adam.layers[0].weights)
    assert not np.allclose(model_mom.layers[0].weights, model_adam.layers[0].weights)

    print("  ✓ Optimizer tests passed")


# ============================================================
# TRAINING LOOP TEST
# ============================================================
def test_training():
    from neural_network import NeuralNetwork
    from helper_functions import one_hot_encode

    print("Testing Training Loop...")

    np.random.seed(42)
    X = np.random.randn(120, 10)
    y = np.random.randint(0, 3, 120)
    y_onehot = one_hot_encode(y, 3)

    model = NeuralNetwork([10, 8, 3], ['relu', 'softmax'], learning_rate=0.1, seed=42)

    hist = model.train(X, y_onehot, epochs=15, batch_size=32, verbose=False)

    assert hist["train_loss"][-1] < hist["train_loss"][0], "Training loss didn't decrease"
    assert hist["train_acc"][-1] >= hist["train_acc"][0], "Training accuracy didn't improve"

    print("  ✓ Training loop test passed")


# ============================================================
# UTILITY TESTS
# ============================================================
def test_utils():
    from helper_functions import one_hot_encode, normalize_data, compute_confusion_matrix, compute_metrics

    print("Testing Utilities...")

    y = np.array([0, 1, 2, 1])
    y1 = one_hot_encode(y, 3)
    assert y1.shape == (4, 3)
    assert np.all(y1.sum(axis=1) == 1)

    X = np.random.randn(100, 5) * 3 + 10
    X2 = np.random.randn(40, 5) * 3 + 10
    Xn, X2n, mean, std = normalize_data(X, X2)
    assert np.abs(np.mean(Xn)) < 0.1
    assert np.abs(np.std(Xn) - 1.0) < 0.1

    # confusion matrix
    yt = np.array([0, 1, 2, 1, 0])
    yp = np.array([0, 1, 2, 0, 1])
    cm = compute_confusion_matrix(yt, yp, 3)
    assert cm.shape == (3, 3)

    metrics = compute_metrics(yt, yp, 3)
    assert "accuracy" in metrics
    assert "precision" in metrics

    print("  ✓ Utility tests passed")


# ============================================================
# RUN ALL TESTS
# ============================================================
def run_all_tests():
    print("=" * 60)
    print("Running Unit Tests for Neural Network Implementation")
    print("=" * 60)

    try:
        test_activation_functions()
        test_layer()
        test_loss_functions()
        test_gradient_numerical()
        test_optimizers()
        test_training()
        test_utils()

        print("\n✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print("\n✗ TEST FAILED:", e)
        return False
    except Exception as e:
        print("\n✗ ERROR:", e)
        return False


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)

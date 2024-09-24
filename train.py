import numpy as np
from models import forward_propagation, sigmoid_derivative
from utils import compute_cost, compute_accuracy

# Backpropagation and parameter update
def train(X_train, Y_train, X_test, Y_test, W1, b1, W2, b2, epochs, learning_rate):
    train_accuracy = []
    test_accuracy = []

    for i in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)

        # Compute cost
        cost = compute_cost(A2, Y_train)

        # Backpropagation
        dZ2 = A2 - Y_train
        dW2 = np.dot(dZ2, A1.T) / X_train.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X_train.shape[1]

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * sigmoid_derivative(Z1)
        dW1 = np.dot(dZ1, X_train.T) / X_train.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X_train.shape[1]

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Track accuracy
        if i % 100 == 0:
            train_accuracy.append(compute_accuracy(A2, Y_train))
            Z1_test, A1_test, Z2_test, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
            test_accuracy.append(compute_accuracy(A2_test, Y_test))
            print(f"Epoch {i}: Cost = {cost:.4f}, Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_accuracy[-1]*100:.2f}%")

    return W1, b1, W2, b2, train_accuracy, test_accuracy

import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Softmax activation function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Convert labels to one-hot encoding
def one_hot_encoding(Y, num_classes):
    one_hot = np.zeros((num_classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot

# Compute cost with correct indexing
def compute_cost(A2, Y):
    m = Y.shape[1]
    log_probs = np.multiply(Y, np.log(A2))
    cost = - np.sum(log_probs) / m
    return cost

# Compute accuracy
def compute_accuracy(A2, Y):
    predictions = np.argmax(A2, axis=0)
    return np.mean(predictions == Y)

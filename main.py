import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from train import train
from models import visualize_network

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize the data
train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

# Set the neural network parameters
input_size = 28 * 28
hidden_size = 64
output_size = 10
epochs = 2000
learning_rate = 0.5

# Train the model and track accuracy
W1, b1, W2, b2, train_accuracy, test_accuracy = train(train_images.T, train_labels, test_images.T, test_labels, input_size, hidden_size, output_size, epochs, learning_rate)

# Plot accuracy over epochs
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize the neural network architecture
visualize_network(input_size, hidden_size, output_size)

# Show an example input image and the corresponding prediction
def show_prediction(W1, b1, W2, b2, X, y):
    from models import forward_propagation
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    prediction = np.argmax(A2, axis=0)

    plt.imshow(X.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Label: {prediction[0]}, True Label: {y}')
    plt.show()

# Test with a sample image from the test set
show_prediction(W1, b1, W2, b2, test_images[0].reshape(-1, 1), test_labels[0])

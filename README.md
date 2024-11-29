# ScratchThatNeuro-Neural-Network-from-scratch-No-Tensorflow
This project implements a simple feed-forward neural network from scratch to classify digits from the MNIST dataset. The network is trained using backpropagation with the sigmoid activation function and softmax for the output layer.

https://github.com/user-attachments/assets/0f707c6d-dae5-44fa-841c-9d63ecb3a76a

![NN Confusion Matrix](https://github.com/user-attachments/assets/cf337f9a-c08c-472b-9a89-5ee76549b392)

## Features
- Achieves 94.42% accuracy on the MNIST dataset.
- Custom implementation of forward and backward propagation
- Cost calculation and accuracy tracking
- Visualization of neural network architecture and accuracy over time
- MNIST dataset used for training and testing
  
![Neural Network Train VS Test Accuracy](https://github.com/user-attachments/assets/9b9d28de-933c-4f98-bfb0-750c4cf82102)

## Usage
### Prerequisites
Make sure you have the following installed:
- `numpy`
- `matplotlib`
- `keras`
### Dataset:
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

### Running the Model
You can run the model with:
```bash
python main.py

### Results
Training Accuracy: 94.42%.
Test Accuracy: Comparable performance, indicating strong generalization on unseen data.
Dataset: The project uses the MNIST dataset of handwritten digits, with 60,000 training and 10,000 testing samples.


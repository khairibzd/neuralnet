# Neural Network Implementation

This project is a Python-based implementation of a fully connected feedforward neural network from scratch, using mini-batch stochastic gradient descent and backpropagation for training. The network can be trained to recognize patterns in data, such as images and classification tasks.

## Features

- Customizable neural network architecture, allowing any number of layers and neurons
- Efficient weight and bias initialization
- Mini-batch Stochastic Gradient Descent (SGD) for training
- Backpropagation for gradient calculation
- Evaluation function to test model accuracy on test data

## Requirements

This project requires Python 3.6+ and the following Python packages:
- `numpy` for numerical calculations

Install dependencies with:

```bash
pip install -r requirements.txt


Code Overview
1. Network Initialization
The neural network is initialized with a specified architecture through the sizes parameter, a list containing the number of neurons in each layer. For example, [2, 3, 1] creates a network with three layers: an input layer with 2 neurons, a hidden layer with 3 neurons, and an output layer with 1 neuron.

def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


2. Feedforward
The feedforward function calculates the output of the network for a given input by passing the data through each layer and applying the activation function.

def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a


3. Stochastic Gradient Descent (SGD)
The SGD function performs mini-batch gradient descent, shuffling the training data and creating mini-batches for each epoch. The network's performance can also be evaluated on test data after each epoch.

def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    # Training loop over epochs

4. Backpropagation
The backprop function computes the gradient of the cost function, layer by layer, to adjust weights and biases. This function uses the chain rule to propagate the error backward from the output layer to the input layer.

def backprop(self, x, y):
    # Compute the gradient of cost function

5. Evaluation
The evaluate function measures the network's accuracy on test data by counting the correct predictions. It assumes the network output is the index of the neuron with the highest activation in the final layer.

def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

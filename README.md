# Handwritten Digit Recognition Using Neural Networks

This project aims to train a neural network to recognize handwritten digits. The goal is to understand the mathematical principles behind how a neural network functions, without using any libraries that simplify the construction of neural networks, such as TensorFlow or PyTorch.

## Data
The dataset used is [MNIST](http://yann.lecun.com/exdb/mnist/), which contains 60,000 training images and 10,000 test images of handwritten digits.

## Preprocessing
The images were read and converted into NumPy arrays. The data was then split into a development set (1,000 examples) and a training set (59,000 examples). The class labels were converted into a one-hot encoded matrix.

## Neural Network
The neural network consists of an input layer with 784 neurons, each representing one of the pixels of each image in the dataset, a hidden layer with 10 neurons, and an output layer with 10 neurons. The ReLU activation function was used in the hidden layer, and the softmax function was used in the output layer. The optimization algorithm used was gradient descent.

## Training
The neural network was trained with the training data using the backpropagation technique to adjust the weights. The learning rate was set to 0.2, and the number of iterations was set to 500 (If you want to test different values, simply change the values in the `gradient_descent` function on line 135 of the `digitRecognizer.py` file).

## Results
The accuracy of the neural network on the development set was approximately 85%.

## References
- [Andrew Ng's Neural Networks and Deep Learning Course on Coursera](https://www.coursera.org/learn/neural-networks-deep-learning)
- [MNIST Homepage](http://yann.lecun.com/exdb/mnist/)

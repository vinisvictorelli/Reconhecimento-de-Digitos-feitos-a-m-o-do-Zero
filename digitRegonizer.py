import numpy as np # linear algebra for calculations
import pandas as pd 
from matplotlib import pyplot as plt

data = pd.read_csv("data/train.csv") # reads the CSV file containing the dataset
data = np.array(data)

m, n = data.shape # obtains the number of examples (m) and features (n)
np.random.shuffle(data)

'''
Splits the data into development and training sets
'''
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

'''
Function to initialize neural network parameters
'''
def initialize_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

'''
Defines the ReLU activation function
'''
def ReLU(Z):
    return np.maximum(Z, 0)

'''
Softmax function is used to convert a vector of numbers into a probability distribution,
where each element of the distribution represents the probability of belonging to a particular class.
'''
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

'''
Computes the activations of the hidden and output layers using the provided weight and bias matrices.
'''
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

'''
The derivative of the ReLU function is important in optimization techniques like backpropagation,
which is used to adjust the weights of the neural network during training.
'''
def ReLU_derivative(Z):
    return Z > 0

'''
Converts class labels into a matrix with one-hot encoding.
'''
def one_hot_encoder(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

'''
Computes the derivatives of the weights and biases of the hidden and output layers 
using the derivatives of the ReLU and softmax activation functions.
'''  
def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot_encoder(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

'''
Updates the weights and biases of the network using the learning rate and the previously computed derivatives.
'''
def update_parameters(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * dB1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * dB2
    return W1, W2, b1, b2

def make_predictions(A2):
    return np.argmax(A2, 0)

def calculate_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
    
'''
Trains the network using the training data and returns the trained weights and biases.
'''
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initialize_parameters()
    print(X.shape)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, dB1, dW2, dB2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, W2, b1, b2 = update_parameters(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha)
        if (i % 10 == 0):
            print('iteration: ', i)
            print('accuracy: ', calculate_accuracy(make_predictions(A2), Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.2)

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = make_predictions(A2)
    return predictions

def test_predictions(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = predict(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_predictions(100, W1, b1, W2, b2)
test_predictions(1, W1, b1, W2, b2)
test_predictions(2, W1, b1, W2, b2)
test_predictions(3, W1, b1, W2, b2)

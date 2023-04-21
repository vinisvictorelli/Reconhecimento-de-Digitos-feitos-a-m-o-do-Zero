import numpy as np # linear algebra
import pandas as pd 
from matplotlib import pyplot as plt

data = pd.read_csv("digit-recognizer/train.csv") #lê o arquivo CSV contendo os dados do dataset
data = np.array(data)

m,n = data.shape #obtém a quantidade de exemplos (m) e de atributos (n)
np.random.shuffle(data)

'''
Divide os dados em conjunto de desenvolvimento e treinamento
'''
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

'''
Função para inicializar os parâmetros da rede neural
'''
def iniciar_parametros():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2

'''
Define a função de ativação ReLU
'''
def ReLU(Z):
    return np.maximum(Z,0)

'''
Função softmax é usada para transformar um vetor de números em uma 
distribuição de probabilidades, onde cada elemento da distribuição 
representa a probabilidade de pertencer a uma determinada classe.
'''
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

'''
Calcula as ativações das camadas oculta e de saída 
usando as matrizes de pesos e bias fornecidas.
'''
def forwardProp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

'''
A derivada da função ReLU é importante em técnicas de 
otimização como o backpropagation, que é usado para 
ajustar os pesos da rede neural durante o treinamento. 
Durante o backpropagation, a derivada da função de ativação 
é usada para calcular os gradientes dos pesos da rede.
'''
def derivada_ReLU(Z):
    return Z > 0

'''
Converte os rótulos da classe em uma matriz com one-hot encoding.
'''
def hotEncoder(Y):
    hot_encode_Y = np.zeros((Y.size,Y.max() + 1))
    hot_encode_Y[np.arange(Y.size),Y] = 1
    hot_encode_Y = hot_encode_Y.T
    return hot_encode_Y

'''
Calcula as derivadas dos pesos e bias da camada oculta e de saída 
usando as derivadas da função de ativação ReLU e softmax.
'''  
def backPropagation(Z1,A1,Z2,A2,W1,W2,X,Y):
    m = Y.size
    hot_encode_Y = hotEncoder(Y)
    dZ2 = A2 - hot_encode_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2)* derivada_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1,dB1,dW2,dB2

'''
Atualiza os pesos e bias da rede usando a 
taxa de aprendizado e as derivadas calculadas anteriormente.
'''
def atualizar_parametros(W1,b1,W2,b2,dW1,dB1,dW2,dB2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * dB1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * dB2
    return W1,W2,b1,b2

def predicoes(A2):
    return np.argmax(A2,0)

def definir_acuracia(predicoes,Y):
    print(predicoes,Y)
    return np.sum(predicoes == Y) / Y.size
    
'''
Treina a rede usando os dados de treinamento e retorna 
os pesos e bias treinados.
'''
def gradient_descent(X,Y,iterations,alpha):
    W1,b1,W2,b2 = iniciar_parametros()
    print(X.shape)
    for i in range(iterations):
        Z1,A1,Z2,A2 = forwardProp(W1,b1,W2,b2,X)
        dW1,dB1,dW2,dB2 = backPropagation(Z1,A1,Z2,A2,W1,W2,X,Y)
        W1,W2,b1,b2 = atualizar_parametros(W1,b1,W2,b2,dW1,dB1,dW2,dB2,alpha)
        if (i % 10 == 0 ):
            print('iteracao: ', i)
            print('acuracia: ', definir_acuracia(predicoes(A2),Y))
    return W1,b1,W2,b2

W1,b1,W2,b2 = gradient_descent(X_train,Y_train,500,0.2)

def fazer_previsoes(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    previsoes = predicoes(A2)
    return previsoes

def testar_previsoes(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = fazer_previsoes(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

testar_previsoes(100, W1, b1, W2, b2)
testar_previsoes(1, W1, b1, W2, b2)
testar_previsoes(2, W1, b1, W2, b2)
testar_previsoes(3, W1, b1, W2, b2)


# Reconhecimento de dígitos manuscritos usando redes neurais

Este projeto tem como objetivo treinar uma rede neural a reconhecer dígitos manuscritos. Além disso, o objetivo é entender matematicamente como uma rede neural funciona, portanto não foi uso nenhuma biblioteca para facilicar a construção da rede neural como tensorflow ou pytorch.

## Dados
O dataset usado é o [MNIST](http://yann.lecun.com/exdb/mnist/), contendo 60.000 imagens de treinamento e 10.000 imagens de teste de dígitos manuscritos. 

## Pré-processamento
As imagens foram lidas e convertidas em arrays NumPy. Em seguida, os dados foram divididos em um conjunto de desenvolvimento (1000 exemplos) e um conjunto de treinamento (59.000 exemplos). Os rótulos de classe foram convertidos em matriz one-hot encoding.

## Rede Neural
A rede neural consiste em uma camada de entrada com 784 neurônios, cada um representando um dos pixels de cada imagem no dataset, uma camada oculta com 10 neurônios e uma camada de saída com 10 neurônios (). A função de ativação ReLU foi usada na camada oculta e a função softmax foi usada na camada de saída. O algoritmo de otimização usado foi o gradient descent.

## Treinamento
A rede neural foi treinada com os dados de treinamento usando a técnica de backpropagation para ajustar os pesos. A taxa de aprendizado foi definida como 0,2 e o número de iterações foi definido como 500 (Caso queira testar valores diferentes, basta mudar os valores na função gradient_descent na linha 135 do arquivo digitRecognizer.py).

## Resultados
A acurácia da rede neural no conjunto de desenvolvimento foi de ~85%

## Referências
- [Curso de Redes Neurais Artificiais do Andrew Ng no Coursera](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Página do MNIST](http://yann.lecun.com/exdb/mnist/)

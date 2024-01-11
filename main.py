import numpy as np
import pandas as pd

#FUNKCJE AKTYWACJI
# Funkcja sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji sigmoidalnej
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#funkcja tangensa hiperbolicznego
def tanh_function(x):
    return np.tanh(x)
#pochodna funkcji tangensa hiperbolicznego
def tanh_derivative(x):
    tanh_x = tanh_function(x)
    return 1 - tanh_x**2

#funkcja progowa
def relu(x):
    return np.maximum(0, x)
#pochodna funkcji progowej
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

df = pd.read_csv('Data/Cardiovascular_Disease_Dataset.csv')
X = df.iloc[:, 1:-1].values  #Wyrzucenie pierwszej zmiennej, ponieważ to id pacjenta
Y = df.iloc[:, -1].values  # Wyodrębnienie zmiennej objaśnianej

X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(-1, 1)

#normalizacja danych
for i in range(np.shape(X)[1]):
    X[:, i] = (X[:, i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i]))

training_X = X[:800]
training_Y = Y[:800]

testing_X = X[800:]
testing_Y = Y[800:]

#Funkcja zwraca losowe wartości wag i biasów dla kolejnych warstw ukrytych oraz dla warstwy wyjścia
def create_layers(num_of_neurons):
    list_weights = []
    list_bias = []
    for i in range(1,len(num_of_neurons)):
        list_weights.append(np.random.uniform(size=(num_of_neurons[i-1], num_of_neurons[i])))
        list_bias.append(np.random.uniform(size=(1, num_of_neurons[i])))
    return list_weights, list_bias

#Trening sieci neuronowej składa się z czterech etapów: propagacja w przód, obliczenie błędu,
# wsteczna propagacja błędu oraz aktualizacja wag i biasów

#Propagacja w przód
#Funkcja jako parametr przyjmuje listę wag i biasów utworzoną przez funkcję create_layers()
def forward_propagation(layers):
    activation = []
    output = []
    activation.append(np.dot(training_X, layers[0][0])+layers[1][0])
    output.append(sigmoid(activation[0]))
    for i in range(1, len(layers[0])):
        activation.append(np.dot(output[i-1], layers[0][i])+layers[1][i])
        output.append(sigmoid(activation[i]))

    return activation, output

# Obliczenie błędu  i wsteczna propagacja
#Funkcja zwraca błąd dla każdej warstwy oraz iloczyn błędu i pochodnej funkcji
def backpropagation(output, weights):
    error = []
    derrivative = []
    error.append(training_Y - output[-1]**2)
    derrivative.append(error * sigmoid_derivative(output[-1]))
    for i, j in zip(range(len(layers[0]) - 1, -1, -1), range(0, len(layers[0]) - 1)):
        error.append(derrivative[j].dot(weights[i].T))
        derrivative.append(error[j + 1] * sigmoid_derivative(output[i - 1]))

    return error, derrivative

# Aktualizacja wag i biasu
def actualisation(layers, fp, back):
    for i, j in zip(range(len(layers[0])-1, 0, -1), range(0, len(layers[0])-1)):
        a,b = fp[1][i-1].T.dot(back[1][j]).shape[0], fp[1][i-1].T.dot(back[1][j]).shape[2]
        layers[0][i] += fp[1][i-1].T.dot(back[1][j]).reshape(a,b) * learning_rate
        c, d = back[1][j].shape[1], back[1][j].shape[2]
        layers[1][i] += np.sum(back[1][j].reshape(c, d), axis=0, keepdims=True) * learning_rate
    c, d = back[1][-1].shape[1], back[1][-1].shape[2]
    layers[0][0] += training_X.T.dot((back[1][-1]).reshape(c, d)) * learning_rate
    layers[1][0] += np.sum(back[1][-1].reshape(c, d), axis=0, keepdims=True) * learning_rate

learning_rate = 0.1
epochs = 10000
#Testujemy dla dwóch warstw ukrytych o liczbie neuronów 4 i 3
layers = create_layers([12, 4, 3, 1])

for epoch in range(epochs):
    fp = forward_propagation(layers)
    back = backpropagation(fp[1], layers[0])
    actualisation(layers, fp, back)


# Wyświetlenie wyników po treningu
print("Wagi warstwy ukrytej 1 po treningu:")
print(layers[0][0])
print("\nBias warstwy ukrytej 1 po treningu:")
print(layers[1][0])
print("\nWagi warstwy ukrytej 2 po treningu:")
print(layers[0][1])
print("\nBias warstwy ukrytej 2 po treningu:")
print(layers[1][1])
print("\nWagi warstwy wyjściowej po treningu:")
print(layers[0][2])
print("\nBias warstwy wyjściowej po treningu:")
print(layers[1][2])

# Testowanie sieci neuronowej


def testing(layers):
    activation = []
    output = []
    activation.append(np.dot(testing_X, layers[0][0])+layers[1][0])
    output.append(sigmoid(activation[0]))
    for i in range(1, len(layers[0])):
        activation.append(np.dot(output[i-1], layers[0][i])+layers[1][i])
        output.append(sigmoid(activation[i]))

    return activation, output

print(testing(layers)[1][-1])


if __name__ == '__main__':
    print("\nWyniki po testowaniu:")
    print(testing(layers)[1][-1])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

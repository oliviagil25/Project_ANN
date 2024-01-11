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
def forward_propagation(weights, biases, function):
    activation = []
    output = []
    activation.append(np.dot(training_X, weights[0])+biases[0])
    output.append(function(activation[0]))
    for i in range(1, len(weights)):
        activation.append(np.dot(output[i-1], weights[i])+biases[i])
        output.append(function(activation[i]))

    return output

# Obliczenie błędu  i wsteczna propagacja
#Funkcja zwraca błąd dla każdej warstwy oraz iloczyn błędu i pochodnej funkcji
def backpropagation(output, weights, f_derivative):
    error = []
    derivative = []
    error.append(training_Y - output[-1]**2)
    derivative.append(error * f_derivative(output[-1]))
    for i, j in zip(range(len(weights) - 1, -1, -1), range(0, len(weights) - 1)):
        error.append(derivative[j].dot(weights[i].T))
        derivative.append(error[j + 1] * f_derivative(output[i - 1]))

    return derivative

# Aktualizacja wag i biasu
def actualisation(weights, biases, fp, back):
    for i, j in zip(range(len(weights)-1, 0, -1), range(0, len(weights)-1)):
        a,b = fp[i-1].T.dot(back[j]).shape[0], fp[i-1].T.dot(back[j]).shape[2]
        weights[i] += fp[i-1].T.dot(back[j]).reshape(a,b) * learning_rate
        c, d = back[j].shape[1], back[j].shape[2]
        biases[i] += np.sum(back[j].reshape(c, d), axis=0, keepdims=True) * learning_rate
    c, d = back[-1].shape[1], back[-1].shape[2]
    weights[0] += training_X.T.dot((back[-1]).reshape(c, d)) * learning_rate
    biases[0] += np.sum(back[-1].reshape(c, d), axis=0, keepdims=True) * learning_rate

learning_rate = 0.1
epochs = 10000
#Testujemy dla dwóch warstw ukrytych o liczbie neuronów 4 i 3
weights, biases = create_layers([12, 4, 3, 1])

for epoch in range(epochs):
    fp = forward_propagation(weights, biases, sigmoid)
    back = backpropagation(fp, weights, sigmoid_derivative)
    actualisation(weights, biases, fp, back)


# Wyświetlenie wyników po treningu
print("Wagi warstwy ukrytej 1 po treningu:")
print(weights[0])
print("\nBias warstwy ukrytej 1 po treningu:")
print(biases[0])
print("\nWagi warstwy ukrytej 2 po treningu:")
print(weights[1])
print("\nBias warstwy ukrytej 2 po treningu:")
print(biases[1])
print("\nWagi warstwy wyjściowej po treningu:")
print(weights[2])
print("\nBias warstwy wyjściowej po treningu:")
print(biases[2])


# Funkcja porównująca wyniki
def compare_results(predictions):
    threshold = 0.5
    predicted_classes = np.where(predictions >= threshold, 1, 0)
    accuracy = np.mean(predicted_classes == testing_Y)
    return accuracy

# Testowanie sieci neuronowej
def testing(weights, biases):
    activation = []
    output = []
    activation.append(np.dot(testing_X, weights[0])+biases[0])
    output.append(sigmoid(activation[0]))
    for i in range(1, len(weights)):
        activation.append(np.dot(output[i-1], weights[i])+biases[i])
        output.append(sigmoid(activation[i]))
    accuracy = compare_results(output[-1])
    return output, accuracy



if __name__ == '__main__':
    print("\nWyniki po testowaniu:")
    print(testing(weights, biases)[0][-1])
    test_accuracy = testing(weights, biases)[1]
    print("Dokładność testowania: {:.2%}".format(test_accuracy))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

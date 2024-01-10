import numpy as np
import pandas as pd

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

trening_X = X[:800]
trening_Y = Y[:800]

testing_X = X[800:]
testing_Y = Y[800:]


# Inicjalizacja wag i biasu losowymi wartościami
#ODTĄD
np.random.seed(1)
input_neurons = 12
hidden1_neurons = 4  # Liczba neuronów w pierwszej warstwie ukrytej
hidden2_neurons = 3  # Liczba neuronów w drugiej warstwie ukrytej
output_neurons = 1

hidden1_weights = np.random.uniform(size=(input_neurons, hidden1_neurons))
hidden1_bias = np.random.uniform(size=(1, hidden1_neurons))
hidden2_weights = np.random.uniform(size=(hidden1_neurons, hidden2_neurons))
hidden2_bias = np.random.uniform(size=(1, hidden2_neurons))
output_weights = np.random.uniform(size=(hidden2_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

#DOTĄD JEST DO WYWALENIA BO ZROBIŁ TO CHAT A NIŻEJ ZROBIŁAM TO SAMO TYLKO ŻE DLA RÓŻNEJ LICZBY
#WARSTW I NEURONÓW, WIĘC JAK POTEM ROBICIE COŚ TO DOBRZE ŻEBY ODWOŁYWAŁO SIĘ DO TEGO CO ZROBIŁAM NIŻEJ
#num_neurons jest wektorem mówiącym ile w każdej warstwie jest neuronów
def create_layers(num_of_neurons):
    list_weights = []
    list_bias = []
    for i in range(1,len(num_of_neurons)):
        list_weights.append(np.random.uniform(size=(num_of_neurons[i-1], num_of_neurons[i])))
        list_bias.append(np.random.uniform(size=(1, num_of_neurons[i])))
    return list_weights, list_bias

#Podajemy ile jest neuronów w której warstwie a zarazem ile jest warstw(len(layers))
#W naszym przypadku ważne, żeby pierwsza warstwa miała 12 neuronów i ostatnia 1 a reszte będzie trzeba testować
layers = create_layers([12,4,3,1])


#to jest taki wskaźnik jak mocno zmieniamy wagi przy każdej iteracji
learning_rate = 0.1
#to jest właściwie liczba iteracji przy trenowaniu
epochs = 10000


# Trening sieci neuronowej za pomocą wstecznej propagacji błędu
def forward_propagation(X, layers):
    activation = []
    output = []
    activation.append(np.dot(X, layers[0][0])+layers[1][0])
    output.append(sigmoid(activation[0]))
    for i in range(1, len(layers[0])):
        activation.append(np.dot(output[i-1], layers[0][i])+layers[1][i])
        output.append(sigmoid(activation[i]))

    return activation, output

fp = forward_propagation(X, layers)

# Obliczenie błędu
e = (Y - output[-1]) ** 2

# output to fp[1] a weights to layers[0]
def backpropagation(output, weights):
    error = []
    derrivative = []
    error.append((Y - output[-1]) ** 2)
    derrivative.append(error * d_sigmoid(output[-1]))
    for i, j in zip(range(len(layers[0] - 1, -1, -1)), range(0, len(layers[0]) - 1)):
        error.append(derrivative[j].dot(weights[i].T))
        derrivative.append(error[j + 1] * d_sigmoid(output[i - 1]))

    return error, derrivative

back = backpropagation(fp[1], layers[0])

# Aktualizacja wag i biasu
def actualisation(layers, fp, back):
    for i, j in zip(range(len(layers) - 1, 0, -1), range(0, len(layers) - 1)):
        layers[0][i] += fp[1][i - 1].T.dot(back[1][j]) * learning_rate
        layers[1][i] += np.sum(back[1][j], axis=0, keepdims=True) * learning_rate
    layers[0][0] += X.T.dot(back[1][-1]) * learning_rate
    layers[1][0] += np.sum(back[1][-1], axis=0, keepdims=True) * learning_rate


# Trening sieci neuronowej za pomocą wstecznej propagacji błędu
for epoch in range(epochs):
    # Forward propagation
    hidden1_layer_activation = np.dot(X, hidden1_weights)
    hidden1_layer_activation += hidden1_bias
    hidden1_layer_output = sigmoid(hidden1_layer_activation)

    hidden2_layer_activation = np.dot(hidden1_layer_output, hidden2_weights)
    hidden2_layer_activation += hidden2_bias
    hidden2_layer_output = sigmoid(hidden2_layer_activation)

    output_layer_activation = np.dot(hidden2_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Obliczenie błędu
    error = Y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden2_layer = d_predicted_output.dot(output_weights.T)
    d_hidden2_layer = error_hidden2_layer * sigmoid_derivative(hidden2_layer_output)

    error_hidden1_layer = d_hidden2_layer.dot(hidden2_weights.T)
    d_hidden1_layer = error_hidden1_layer * sigmoid_derivative(hidden1_layer_output)

    # Aktualizacja wag i biasu
    output_weights += hidden2_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden2_weights += hidden1_layer_output.T.dot(d_hidden2_layer) * learning_rate
    hidden2_bias += np.sum(d_hidden2_layer, axis=0, keepdims=True) * learning_rate
    hidden1_weights += X.T.dot(d_hidden1_layer) * learning_rate
    hidden1_bias += np.sum(d_hidden1_layer, axis=0, keepdims=True) * learning_rate

# Wyświetlenie wyników po treningu
print("Wagi warstwy ukrytej 1 po treningu:")
print(hidden1_weights)
print("\nBias warstwy ukrytej 1 po treningu:")
print(hidden1_bias)
print("\nWagi warstwy ukrytej 2 po treningu:")
print(hidden2_weights)
print("\nBias warstwy ukrytej 2 po treningu:")
print(hidden2_bias)
print("\nWagi warstwy wyjściowej po treningu:")
print(output_weights)
print("\nBias warstwy wyjściowej po treningu:")
print(output_bias)

# Testowanie sieci neuronowej
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden1_layer_activation = np.dot(test_input, hidden1_weights)
hidden1_layer_activation += hidden1_bias
hidden1_layer_output = sigmoid(hidden1_layer_activation)

hidden2_layer_activation = np.dot(hidden1_layer_output, hidden2_weights)
hidden2_layer_activation += hidden2_bias
hidden2_layer_output = sigmoid(hidden2_layer_activation)

output_layer_activation = np.dot(hidden2_layer_output, output_weights)
output_layer_activation += output_bias
predicted_output = sigmoid(output_layer_activation)


if __name__ == '__main__':
    print("\nWyniki po testowaniu:")
    print(predicted_output)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

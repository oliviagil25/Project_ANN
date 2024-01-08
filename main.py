import numpy as np
import pandas as pd

# Funkcja sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Pochodna funkcji sigmoidalnej
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Utworzenie zestawu danych treningowych (operacja XOR)
inputs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

expected_output = np.array([[0], [1], [1], [0]])

df = pd.read_csv('Data/Cardiovascular_Disease_Dataset.csv')
X = df.iloc[:, 1:-1].values  # Pierwsze 12 kolumn jako zmienne wejściowe
Y = df.iloc[:, -1].values  # Trzynasta kolumna jako zmienna wyjściowa

X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(-1, 1)

trening_X = X[:600]
trening_Y = Y[:600]

testing_X = X[600:800]
testing_Y = Y[600:800]

# Inicjalizacja wag i biasu losowymi wartościami
np.random.seed(1)
input_neurons = 12  # Liczba neuronów w warstwie wejściowej
hidden_neurons = 4
output_neurons = 1

hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 10000

# Trening sieci neuronowej za pomocą wstecznej propagacji błędu
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(X, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Obliczenie błędu
    error = Y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Aktualizacja wag i biasu
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Wyświetlenie wyników po treningu
print("Wagi warstwy ukrytej po treningu:")
print(hidden_weights)
print("\nBias warstwy ukrytej po treningu:")
print(hidden_bias)
print("\nWagi warstwy wyjściowej po treningu:")
print(output_weights)
print("\nBias warstwy wyjściowej po treningu:")
print(output_bias)

# Testowanie sieci neuronowej
test_input = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
hidden_layer_activation = np.dot(testing_X, hidden_weights)
hidden_layer_activation += hidden_bias
hidden_layer_output = sigmoid(hidden_layer_activation)

output_layer_activation = np.dot(hidden_layer_output, output_weights)
output_layer_activation += output_bias
predicted_output = sigmoid(output_layer_activation)


if __name__ == '__main__':
    print("\nWyniki po testowaniu:")
    print(predicted_output)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

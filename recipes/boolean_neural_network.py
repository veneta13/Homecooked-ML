import numpy as np


class BooleanNeuralNetwork:
    def __init__(self, layers_sizes, learning_rate=0.1):
        self.layers = np.array(layers_sizes)
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append(
                np.random.uniform(
                    low=-0.05,
                    high=0.05,
                    size=(self.layers[i], self.layers[i + 1])
                )
            )

            self.biases.append(
                np.random.uniform(
                    low=-0.05,
                    high=0.05,
                    size=self.layers[i + 1])
            )

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def __sigmoid_activation(self, X):
        layer_activation = np.zeros_like(X)
        for index, _ in enumerate(X):
            layer_activation[index] = self.__sigmoid(-X[index])
        return layer_activation

    def __error_output(self, pred_value, true_value):
        return np.dot(
            np.dot(
                pred_value,
                1 - pred_value
            ),
            true_value - pred_value
        )

    def __error_layer(self, pred_value, error, weights):
        weighted_error = np.matmul(error, weights.T)
        return np.dot(
            np.dot(
                pred_value,
                1 - pred_value
            ),
            weighted_error
        )

    def __update_weights(self, idx, prev_activation, error):
        self.weights[idx] += self.learning_rate * np.matmul(
            np.reshape(prev_activation, (-1, 1)),
            np.reshape(error, (1, -1))
        )
        self.biases[idx] += self.learning_rate * error

    def __forward(self, x):
        x = np.array(x)
        output = [x]

        for i in range(len(self.layers) - 1):
            layer_calculation = np.matmul(output[-1], self.weights[i])
            layer_calculation += self.biases[i]

            activated_output = self.__sigmoid_activation(layer_calculation)
            output.append(activated_output)

        return output

    def __backward(self, output, errors_last_layer):
        for i in range(len(self.layers) - 2, -1, -1):
            errors = self.__error_layer(output[i], errors_last_layer, self.weights[i])
            self.__update_weights(i, output[i], errors_last_layer)
            errors_last_layer = errors

    def predict(self, x):
        return self.__forward(x)[-1]

    def train(self, X, y, epochs=50_000, verbose=True):
        for epoch in range(epochs):
            for x_value, y_value in zip(X, y):
                outputs = self.__forward(x_value)
                last_errors = self.__error_output(outputs[-1], y_value)
                self.__backward(outputs, last_errors)

            if verbose and epoch % 1000 == 0:
                print('Epoch: ', epoch)
                print('Weights: ', self.weights)

    def test(self, X, y, verbose=True):
        true_classification = 0

        for x_value, y_value in zip(X, y):
            y_pred = self.predict(x_value)[0]
            if np.round(y_pred) == y_value:
                true_classification += 1

            if verbose:
                print(f'Input: {x_value} \t Expected output: {y_value} \t Output: {y_pred}')

        print("Accuracy: ", true_classification / len(X))

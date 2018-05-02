from numpy import exp, array, random, dot


class NeuralNetwork:
    def __init__(self):
        random.seed(1)

        self.sympatic_weigths = 2 * random.random((3, 1)) - 1

    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_input, training_output, epochs):
        for x in xrange(epochs):
            output = self.predict(training_input)

            error = training_output - output

            adjustment = dot(training_input.T, error * self._sigmoid_derivative(output))

            self.sympatic_weigths += adjustment

    def predict(self, inputs):
        return self._sigmoid(dot(inputs, self.sympatic_weigths))


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    print "Initial weight values"
    print neural_network.sympatic_weigths

    training_set_inputs = array([[0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]])
    training_set_outputs = array([[1, 1, 1, 1]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print "Updated weight values"
    print neural_network.sympatic_weigths

    print neural_network.predict(array([0, 1, 0]))

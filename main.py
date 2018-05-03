from numpy import exp, array, random, dot, mean, abs


class NeuralNetwork:
    def __init__(self, layers):
        random.seed(1)
        self.collected_weights = {}
        self.number_of_layers = 1
        if (layers):
            self.number_of_layers = layers

        self.sympatic_weights = {}
        self.sympatic_weights[0] = (2 * random.random((1, 4)) - 1)
        self.sympatic_weights[1] = (2 * random.random((4, 1)) - 1)

    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_input, training_output, i):
        k0 = training_input
        k1 = self._sigmoid(dot(k0, self.sympatic_weights[0]))
        k2 = self._sigmoid(dot(k1.T, self.sympatic_weights[1]))

        k2_error = training_output - k2

        if i % 500 == 0:
            print "Error:" + str(mean(abs(k2_error)))

        k2_delta = k2_error * self._sigmoid_derivative(k2)

        k1_error = k2_delta.dot(self.sympatic_weights[1].T)

        k1_delta = k1_error * self._sigmoid_derivative(k1)

        self.sympatic_weights[1] += k1.T.dot(k2_delta)
        self.sympatic_weights[0] += k0.T.dot(k1_delta)

    def predict(self, inputs):
        return self._feed_forward(inputs)

    def _feed_forward(self, x):
        output = x
        for i in xrange(self.number_of_layers):
            output = self.collected_weights[i] = self._sigmoid(dot(output, self.sympatic_weights[i]))

        return output


def readAndPrepareData():
    data = []
    resolved_data = {}
    with open('market-price.csv', 'r') as file:
        for line in file:
            data.append(float(line.split(',')[1]))

    for i in xrange(len(data) - 1):
        resolved_data[i] = {"price": data[i + 1], "change": calculateChange(data[i], data[i + 1])}

    return resolved_data


def calculateChange(last, now):
    return (now - last) / last


if __name__ == '__main__':
    neural_network = NeuralNetwork(2)

    data = readAndPrepareData()

    print "Initial weight values"
    print neural_network.sympatic_weights

    training_set_inputs = array([[0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]])
    training_set_outputs = array([[1, 1, 1, 1]]).T

    for i in xrange(1000):
        for j in xrange(len(data) - 5):
            training_set_inputs = array([
                [data[j]["change"]]
                , [data[j + 1]["change"]]
                , [data[j + 2]["change"]]
                , [data[j + 3]["change"]]])

            training_set_outputs = array([[data[j + 4]["change"]]])
            neural_network.train(training_set_inputs, training_set_outputs, i * j)

    print "Updated weight values"
    print neural_network.sympatic_weights

    print neural_network.predict(array([[data[0]["change"]]
                                       , [data[1]["change"]]
                                       , [data[2]["change"]]
                                       , [data[3]["change"]]]))
    print data[4]["change"]

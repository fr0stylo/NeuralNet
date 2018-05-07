from numpy import array, random, dot, mean, abs, tanh


# MASE error
# LSTM time series
# output TANH +

class NeuralNetwork:

    def __init__(self, learning_rate=1.0):

        random.seed(1)
        self.learning_rate = learning_rate
        self.sympatic_weights = {}
        self.sympatic_weights[0] = (2 * random.random((4, 8)) - 1)
        self.sympatic_weights[1] = (2 * random.random((8, 4)) - 1)
        self.sympatic_weights[2] = (2 * random.random((4, 1)) - 1)

    def _tanh(self, x, derivative=False):
        if derivative:
            return 1.0 - tanh(x) ** 2
        return tanh(x)

    def train(self, training_input, training_output, i):
        k0 = training_input
        k1 = self._tanh(dot(k0, self.sympatic_weights[0]))
        k2 = self._tanh(dot(k1, self.sympatic_weights[1]))
        k3 = self._tanh(dot(k2, self.sympatic_weights[2]))

        k3_error = training_output - k3

        if i % 1000 == 0:
            print "Error:" + str(mean(abs(k3_error)))

        k3_delta = k3_error * self._tanh(k3, derivative=True)

        k2_error = k3_delta.dot(self.sympatic_weights[2].T)

        k2_delta = k2_error * self._tanh(k2, derivative=True)

        k1_error = k2_delta.dot(self.sympatic_weights[1].T)

        k1_delta = k1_error * self._tanh(k1, derivative=True)

        self.sympatic_weights[2] += self.learning_rate * k2.T.dot(k3_delta)
        self.sympatic_weights[1] += self.learning_rate * k1.T.dot(k2_delta)
        self.sympatic_weights[0] += self.learning_rate * k0.T.dot(k1_delta)

    def predict(self, inputs):
        return self._feed_forward(inputs)

    def _feed_forward(self, x):
        k0 = x
        k1 = self._tanh(dot(k0, self.sympatic_weights[0]))
        k2 = self._tanh(dot(k1, self.sympatic_weights[1]))
        k3 = self._tanh(dot(k2, self.sympatic_weights[2]))

        return k3


def readAndPrepareData():
    data = []
    resolved_data = {}
    with open('market-price.csv', 'r') as file:
        for line in file:
            data.append(float(line.split(',')[1]))

    for i in xrange(len(data) - 1):
        resolved_data[i] = {"price": data[i + 1], "change": calculateChange(data[i], data[i + 1])}

    resolved_data = {k: v for k, v in resolved_data.iteritems() if v["change"] < 1}
    return resolved_data


def calculateChange(last, now):
    return (now - last) / last


if __name__ == '__main__':
    neural_network = NeuralNetwork(learning_rate=0.0001)

    data = readAndPrepareData()

    print "Initial weight values"
    print neural_network.sympatic_weights

    for i in xrange(10000):
        for j in xrange(len(data) - 55):
            training_set_inputs = array([
                [data[j]["change"]],
                [data[j + 1]["change"]],
                [data[j + 2]["change"]],
                [data[j + 3]["change"]]]).T

            training_set_outputs = array([[data[j + 4]["change"]]])
            neural_network.train(training_set_inputs, training_set_outputs, i * j)

    mases = []
    mae_predictions = []
    mae_dummys = []
    for j in xrange(len(data) - 55, len(data) - 5):
        test_set_intputs = array([
            [data[j]["change"]]
            , [data[j + 1]["change"]]
            , [data[j + 2]["change"]]
            , [data[j + 3]["change"]]]).T

        test_set_outputs = array([[data[j + 4]["change"]]])

        prediction = neural_network.predict(test_set_intputs)

        mae_predictions.append(mean(abs(test_set_outputs - prediction)))
        mae_dummys.append(mean(abs(test_set_outputs - array([[data[j + 3]["change"]]]))))

    mase = sum(mae_predictions) / sum(mae_dummys)

    print "avg MASE:"
    print mase

from numpy import exp, array, random, dot, mean, abs, tanh


# MASE error
# LSTM time series
# output TANH +

class NeuralNetwork:

    def __init__(self, learning_rate=1.0):

        random.seed(1)
        self.learning_rate = learning_rate
        self.sympatic_weights = {}
        self.sympatic_weights[0] = (2 * random.random((4, 4)) - 1)
        self.sympatic_weights[1] = (2 * random.random((4, 1)) - 1)

    def _sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + exp(-x))

    def _tanh(self, x, derivative=False):
        if derivative:
            return 1.0 - tanh(x) ** 2
        return tanh(x)

    def train(self, training_input, training_output, i):
        k0 = training_input
        k1 = self._tanh(dot(k0, self.sympatic_weights[0]))
        k2 = self._tanh(dot(k1, self.sympatic_weights[1]))

        k2_error = training_output - k2

        if i % 1000 == 0:
            print "Error:" + str(mean(abs(k2_error)))

        k2_delta = k2_error * self._tanh(k2, derivative=True)

        k1_error = k2_delta.dot(self.sympatic_weights[1].T)

        k1_delta = k1_error * self._tanh(k1, derivative=True)

        self.sympatic_weights[1] += self.learning_rate * k1.T.dot(k2_delta)
        self.sympatic_weights[0] += self.learning_rate * k0.T.dot(k1_delta)

    def predict(self, inputs):
        return self._feed_forward(inputs)

    def _feed_forward(self, x):
        k0 = x
        k1 = self._tanh(dot(k0, self.sympatic_weights[0]))
        k2 = self._tanh(dot(k1, self.sympatic_weights[1]))

        return k2


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
    neural_network = NeuralNetwork(learning_rate=0.07)

    data = readAndPrepareData()

    print "Initial weight values"
    print neural_network.sympatic_weights

    for i in xrange(1000):
        for j in xrange(len(data) - 55):
            training_set_inputs = array([
                [data[j]["change"]],
                [data[j + 1]["change"]],
                [data[j + 2]["change"]],
                [data[j + 3]["change"]]]).T

            training_set_outputs = array([[data[j + 4]["change"]]])
            neural_network.train(training_set_inputs, training_set_outputs, i * j)

    # count = 0
    # for j in xrange(len(data) - 55, len(data) - 5):
    #     test_set_intput = array([
    #         [data[j]["change"]]
    #         , [data[j + 1]["change"]]
    #         , [data[j + 2]["change"]]
    #         , [data[j + 3]["change"]]]).T
    #
    #     test_set_outputs = array([[data[j + 4]["change"]]]).T
    #
    #     prediction = neural_network.predict(training_set_inputs)
    #     if prediction == test_set_outputs:
    #         count = count + 1
    #
    # print "Teisingai prognozuotu kekis"
    # print count

    print "Updated weight values"
    print neural_network.sympatic_weights

    print "Predicted value"
    print neural_network.predict(array([
        [data[0]["change"]]
        , [data[1]["change"]]
        , [data[2]["change"]]
        , [data[3]["change"]]
    ]).T)

    print "True value"
    print array([[data[4]["change"]]])

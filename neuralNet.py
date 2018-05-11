from numpy import array, random, dot, mean, abs, tanh, floor


class Net():
    def __init__(self, learning_rate=0.1, inputs_count=4):
        random.seed(1)
        self.multiplier = 2
        self.learning_rate = learning_rate
        n = inputs_count * self.multiplier
        if n % 2 != 1:
            n = n + 1
        self.weights = [(2 * random.random((inputs_count, n)) - 1)]
        while n != 1:
            self.weights.append((2 * random.random((int(n), int(n / 2))) - 1))
            n = int(n / 2)

    def _tanh(self, x):
        return tanh(x)

    def _tanh_derivative(self, x):
        return 1.0 - self._tanh(x) ** 2

    def _feed_forward(self, inputs):
        results = [inputs]
        for i in xrange(len(self.weights) - 1):
            results.append(self._tanh(dot(results[-1], self.weights[i])))

        results.append(dot(results[-1], self.weights[-1]))

        return results

    def _back_propagate(self, output, results):
        last_layer_error = output - results[-1]
        last_layer_delta = last_layer_error * self._tanh_derivative(results[-1])
        deltas = [last_layer_delta]
        for idx in xrange(2, len(self.weights)):
            layer_number = idx * -1
            layer_error = deltas[-1].dot(self.weights[layer_number + 1].T)
            deltas.append(layer_error * self._tanh_derivative(results[layer_number]))

        for idx in xrange(1, len(self.weights)):
            layer_number = idx * -1
            self.weights[idx] += self.learning_rate * results[idx].T.dot(deltas[layer_number])

    def train(self, training_input_set, training_output_set, epochs=1000):
        for i in xrange(epochs):
            for input_idx in xrange(len(training_input_set)):
                results = self._feed_forward(training_input_set[input_idx])
                self._back_propagate(training_output_set[input_idx], results)

    def predict(self, inputs):
        return self._feed_forward(inputs)[-1]

    def print_weights(self):
        print self.weights


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
    lr = 0.005
    input_count = 4
    epochs = 1000
    training_set_size = 70
    test_set_size = 30

    neural_network = Net(learning_rate=lr, inputs_count=input_count)

    data = readAndPrepareData()

    print "Initial weight values"
    print neural_network.weights

    training_set_inputs = []
    training_set_outputs = []

    # TODO: input strategy refactor
    for j in xrange((len(data) / 100 * training_set_size) - 4):
        training_set_inputs.append(
            array([
                array([data[j]["change"]]).T,
                array([data[j + 1]["change"]]).T,
                array([data[j + 2]["change"]]).T,
                array([data[j + 2]["change"]]).T,
                # [data[j]["change"], j % 12],
                # [data[j + 1]["change"], j % 12],
                # [data[j + 2]["change"], j % 12],
                # [data[j + 3]["change"], j % 12]
            ]).T
        )

        training_set_outputs.append(array([[data[j + 4]["change"]]]))

    neural_network.train(training_set_inputs, training_set_outputs, epochs)

    mases = []
    mae_predictions = []
    mae_dummys = []

    for j in xrange(len(data) / 100 * training_set_size, len(data) - 4):
        test_set_inputs = array([
            array([data[j]["change"]]).T,
            array([data[j+1]["change"]]).T,
            array([data[j+2]["change"]]).T,
            array([data[j+2]["change"]]).T,
            # [data[j + 1]["change"], j % 12],
            # [data[j + 2]["change"], j % 12],
            # [data[j + 3]["change"], j % 12]
        ]).T

        test_set_outputs = array([[data[j + 4]["change"]]])

        prediction = neural_network.predict(test_set_inputs)
        print prediction
        mae_predictions.append(mean(abs(test_set_outputs - prediction)))
        mae_dummys.append(mean(abs(test_set_outputs - array([[data[j + 3]["change"]]]))))

    mase = sum(mae_predictions) / sum(mae_dummys)

    print "avg MASE:"
    print mase

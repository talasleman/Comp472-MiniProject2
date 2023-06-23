# mlp classifier
import numpy


class InputLayer:
    def __init__(self, inputs):
        self.Inputs = inputs
        self.Outputs = inputs


class HiddenLayer:
    def __init__(self, inputs, numberNodes, weights):
        self.Inputs = inputs
        self.NumberNodes = numberNodes
        self.Weights = weights
        self.Outputs = None


def sigmoid(input):
    return 1 / (1 + numpy.exp(-input))


inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
for input in inputs:
    inp = InputLayer(input)
    hid = HiddenLayer(inp.Outputs, 2, [-1, 1, 1])

    out = HiddenLayer(hid.Outputs, 1, [-1, 1, 1])


def calculateOutputs(inputs, HiddenLayer):
    sigmoid(HiddenLayer.Weights[0] + HiddenLayer.Weights[1]*inputs[0] +)
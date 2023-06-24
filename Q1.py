# mlp classifier
import numpy as np


class InputLayer:
    def __init__(self, inputs):
        self.Inputs = inputs
        self.Outputs = inputs


class HiddenLayer:
    def __init__(self, inputs, numberNodes, weights):
        self.Inputs = inputs
        self.NumberNodes = numberNodes
        self.Weights = weights
        self.Outputs = np.array([0, 0])


class OutputLayer:
    def __init__(self, inputs, numberNodes, weights):
        self.Inputs = inputs
        self.NumberNodes = numberNodes
        self.Weights = weights
        self.Outputs = np.array([0])


def Activation(input):
    if input > 0:
        return 1
    else:
        return 0


def calculate(InputLayer, HiddenLayer):
    return Activation(HiddenLayer.Weights[0] + HiddenLayer.Weights[1]*InputLayer.Outputs[0] +
                      HiddenLayer.Weights[2]*InputLayer.Outputs[1])


def calculateOutputs(inputs, HiddenLayer):
    return Activation(HiddenLayer.Weights[0] + HiddenLayer.Weights[1]*inputs[0] + HiddenLayer.Weights[2]*inputs[1])


inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
Weights = np.array([-1, 1, 1])
def ANDgate(inputs, weights):
    for input in inputs:
        inp = InputLayer(input)
        hid = HiddenLayer(inp.Outputs, 2, weights)
        hid.Outputs[0] = calculate(inp, hid)
        hid.Outputs[1] = calculate(inp, hid)

        out = OutputLayer(hid.Outputs, 1, weights)
        out.Outputs[0] = calculateOutputs(hid.Outputs, out)
        print("The output for " + input + " is: " + out.Outputs[0])

ANDgate(inputs, Weights)
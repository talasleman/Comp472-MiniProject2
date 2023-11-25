# mlp classifier

class InputLayer:
    def __init__(self, inputs):
        self.Inputs = inputs
        self.Outputs = inputs


class HiddenLayer:
    def __init__(self, inputs, numberNodes, weights):
        self.Inputs = inputs
        self.NumberNodes = numberNodes
        self.Weights = weights
        self.Outputs = []


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


# inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
Weights = [-1, 1, 1]

user = []
user.append(input("AND gate inputs: "))
inputs = []
for element in user:
    for char in element:
        if char != " ":
            inputs.append(int(char))

def ANDgate(inputs, weights):

    inp = InputLayer(inputs)
    hid = HiddenLayer(inp.Outputs, 2, weights)
    hid.Outputs.append(calculate(inp, hid))
    hid.Outputs.append(calculate(inp, hid))

    out = HiddenLayer(hid.Outputs, 1, weights)
    out.Outputs.append(calculateOutputs(hid.Outputs, out))
    x = [str(x) for x in inputs]
    print(f"The output for {', '.join(x)} is: {str(out.Outputs[0])}")

ANDgate(inputs, Weights)
# input 0 0, with a space in between

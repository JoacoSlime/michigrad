import random
from enum import Enum, auto

from michigrad.engine import Value


class NeuronType(Enum):
    Linear = auto()
    ReLU = auto()
    Tanh = auto()
    Sigmoid = auto()


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"LinearNeuron({len(self.w)})"


class ReLU(Neuron):
    def __call__(self, x):
        out = super().__call__(x)
        return out.relu()

    def __repr__(self):
        return f"ReLu({len(self.w)})"


class Tanh(Neuron):
    def __call__(self, x):
        out = super().__call__(x)
        return out.tanh()

    def __repr__(self):
        return f"Tanh({len(self.w)})"


class Sigmoide(Neuron):
    def __call__(self, x):
        out = super().__call__(x)
        return out.sigmoide()

    def __repr__(self):
        return f"Sigmoide({len(self.w)})"


class Layer(Module):
    def __init__(self):
        """
        No instanciar manualmente, usar el método new_layer.
        """
        self.neurons: list[Neuron] = []

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    @staticmethod
    def new_layer(nin: int, nout: int, type: NeuronType, **kwargs):
        """Factory Method"""
        layer = Layer()

        neurons = {
            NeuronType.Linear: Neuron,
            NeuronType.ReLU: ReLU,
            NeuronType.Tanh: Tanh,
            NeuronType.Sigmoid: Sigmoide,
        }

        layer.neurons = [neurons[type](nin, **kwargs) for _ in range(nout)]

        return layer


class MLP(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

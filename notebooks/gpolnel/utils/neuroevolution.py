import torch
from math import prod
from copy import deepcopy
from gpolnel.utils.solution import Solution
from gpolnel.utils.inductive_programming import _Function
from gpolnel.utils.utils import phi


def _feedforward_nn(repr_, activation, x, device):
    previous_layer_output = x.clone().detach().to(device)
    weights = repr_[0]
    biases = repr_[1]
    for i_layer in range(len(weights) - 1):
        # Linear output
        Z = torch.matmul(previous_layer_output, weights[i_layer]) + biases[i_layer]
        # Non-linear (activated) output
        previous_layer_output = activation[i_layer](Z)
    # Output layer
    output = torch.matmul(previous_layer_output, weights[i_layer + 1]) + biases[i_layer + 1]

    return output.squeeze() if activation[i_layer + 1] is None else activation[i_layer + 1](output).squeeze()


"""Credit: mostly based on Ilya's excellent implementation here: https://github.com/ikostrikov/pytorch-flows"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import masks


class InverseAutoregressiveFlow(nn.Module):
    """Inverse Autoregressive Flows with LSTM-type update. One block.

    Eq 11-14 of https://arxiv.org/abs/1606.04934
    """

    def __init__(self, num_input, num_hidden, num_context):
        super().__init__()
        self.made = MADE(
            num_input=num_input,
            num_outputs_per_input=2,
            num_hidden=num_hidden,
            num_context=num_context,
        )
        # init such that sigmoid(s) is close to 1 for stability
        self.sigmoid_arg_bias = nn.Parameter(torch.ones(num_input) * 2)
        self.sigmoid = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, input, context=None):
        m, s = torch.chunk(self.made(input, context), chunks=2, dim=-1)
        s = s + self.sigmoid_arg_bias
        sigmoid = self.sigmoid(s)
        z = sigmoid * input + (1 - sigmoid) * m
        return z, -self.log_sigmoid(s)


class FlowSequential(nn.Sequential):
    """Forward pass."""

    def forward(self, input, context=None):
        total_log_prob = torch.zeros_like(input, device=input.device)
        for block in self._modules.values():
            input, log_prob = block(input, context)
            total_log_prob += log_prob
        return input, total_log_prob


class MaskedLinear(nn.Module):
    """Linear layer with some input-output connections masked."""

    def __init__(
        self, in_features, out_features, mask, context_features=None, bias=True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.register_buffer("mask", mask)
        if context_features is not None:
            self.cond_linear = nn.Linear(context_features, out_features, bias=False)

    def forward(self, input, context=None):
        output = F.linear(input, self.mask * self.linear.weight, self.linear.bias)
        if context is None:
            return output
        else:
            return output + self.cond_linear(context)


class MADE(nn.Module):
    """Implements MADE: Masked Autoencoder for Distribution Estimation.

    Follows https://arxiv.org/abs/1502.03509

    This is used to build MAF: Masked Autoregressive Flow (https://arxiv.org/abs/1705.07057).
    """

    def __init__(self, num_input, num_outputs_per_input, num_hidden, num_context):
        super().__init__()
        # m corresponds to m(k), the maximum degree of a node in the MADE paper
        self._m = []
        degrees = masks.create_degrees(
            input_size=num_input,
            hidden_units=[num_hidden] * 2,
            input_order="left-to-right",
            hidden_degrees="equal",
        )
        self._masks = masks.create_masks(degrees)
        self._masks[-1] = np.hstack(
            [self._masks[-1] for _ in range(num_outputs_per_input)]
        )
        self._masks = [torch.from_numpy(m.T) for m in self._masks]
        modules = []
        self.input_context_net = MaskedLinear(
            num_input, num_hidden, self._masks[0], num_context
        )
        self.net = nn.Sequential(
            nn.ReLU(),
            MaskedLinear(num_hidden, num_hidden, self._masks[1], context_features=None),
            nn.ReLU(),
            MaskedLinear(
                num_hidden,
                num_outputs_per_input * num_input,
                self._masks[2],
                context_features=None,
            ),
        )

    def forward(self, input, context=None):
        # first hidden layer receives input and context
        hidden = self.input_context_net(input, context)
        # rest of the network is conditioned on both input and context
        return self.net(hidden)


class Reverse(nn.Module):
    """An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).

    From https://github.com/ikostrikov/pytorch-flows/blob/master/main.py
    """

    def __init__(self, num_input):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_input)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, context=None, mode="forward"):
        if mode == "forward":
            return inputs[:, :, self.perm], torch.zeros_like(
                inputs, device=inputs.device
            )
        elif mode == "inverse":
            return inputs[:, :, self.inv_perm], torch.zeros_like(
                inputs, device=inputs.device
            )
        else:
            raise ValueError("Mode must be one of {forward, inverse}.")

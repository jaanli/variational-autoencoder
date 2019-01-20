"""Credit: mostly based on Ilya's excellent implementation here: https://github.com/ikostrikov/pytorch-flows"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class InverseAutoregressiveFlow(nn.Module):
  """Inverse Autoregressive Flows with LSTM-type update. One block.
  
  Eq 11-14 of https://arxiv.org/abs/1606.04934
  """
  def __init__(self, num_input, num_hidden, num_context):
    super().__init__()
    self.made = MADE(num_input=num_input, num_output=num_input * 2,
                     num_hidden=num_hidden, num_context=num_context)
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
  def __init__(self, in_features, out_features, mask, context_features=None, bias=True):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features, bias)
    self.register_buffer("mask", mask)
    if context_features is not None:
      self.cond_linear = nn.Linear(context_features, out_features, bias=False)

  def forward(self, input, context=None):
    output =  F.linear(input, self.mask * self.linear.weight, self.linear.bias)
    if context is None:
      return output
    else:
      return output + self.cond_linear(context)


class MADE(nn.Module):
  """Implements MADE: Masked Autoencoder for Distribution Estimation.

  Follows https://arxiv.org/abs/1502.03509

  This is used to build MAF: Masked Autoregressive Flow (https://arxiv.org/abs/1705.07057).
  """
  def __init__(self, num_input, num_output, num_hidden, num_context):
    super().__init__()
    # m corresponds to m(k), the maximum degree of a node in the MADE paper
    self._m = []
    self._masks = []
    self._build_masks(num_input, num_output, num_hidden, num_layers=3)
    self._check_masks()
    modules = []
    self.input_context_net = MaskedLinear(num_input, num_hidden, self._masks[0], num_context)
    modules.append(nn.ReLU())
    modules.append(MaskedLinear(num_hidden, num_hidden, self._masks[1], context_features=None))
    modules.append(nn.ReLU())
    modules.append(MaskedLinear(num_hidden, num_output, self._masks[2], context_features=None))
    self.net = nn.Sequential(*modules)

  def _build_masks(self, num_input, num_output, num_hidden, num_layers):
    """Build the masks according to Eq 12 and 13 in the MADE paper."""
    rng = np.random.RandomState(0)
    # assign input units a number between 1 and D
    self._m.append(np.arange(1, num_input + 1))
    for i in range(1, num_layers + 1):
      # randomly assign maximum number of input nodes to connect to
      if i == num_layers:
        # assign output layer units a number between 1 and D
        m = np.arange(1, num_input + 1)
        assert num_output % num_input == 0, "num_output must be multiple of num_input"
        self._m.append(np.hstack([m for _ in range(num_output // num_input)]))
      else:
        # assign hidden layer units a number between 1 and D-1
        self._m.append(rng.randint(1, num_input, size=num_hidden))
        #self._m.append(np.arange(1, num_hidden + 1) % (num_input - 1) + 1)
      if i == num_layers:
        mask = self._m[i][None, :] > self._m[i - 1][:, None]
      else:
        # input to hidden & hidden to hidden
        mask = self._m[i][None, :] >= self._m[i - 1][:, None]
      # need to transpose for torch linear layer, shape (num_output, num_input)
      self._masks.append(torch.from_numpy(mask.astype(np.float32).T))

  def _check_masks(self):
    """Check that the connectivity matrix between layers is lower triangular."""
    # (num_input, num_hidden)
    prev = self._masks[0].t()
    for i in range(1, len(self._masks)):
      # num_hidden is second axis
      prev = prev @ self._masks[i].t()
    final = prev.numpy()
    num_input = self._masks[0].shape[1]
    num_output = self._masks[-1].shape[0]
    assert final.shape == (num_input, num_output)
    if num_output == num_input:
      assert np.triu(final).all() == 0
    else:
      for submat in np.split(final, 
                             indices_or_sections=num_output // num_input,
                             axis=1):
        assert np.triu(submat).all() == 0

  def forward(self, input, context=None):
    # first hidden layer receives input and context
    hidden = self.input_context_net(input, context)
    # rest of the network is conditioned on both input and context
    return self.net(hidden)



class Reverse(nn.Module):
  """ An implementation of a reversing layer from
  Density estimation using Real NVP
  (https://arxiv.org/abs/1605.08803).

  From https://github.com/ikostrikov/pytorch-flows/blob/master/main.py
  """

  def __init__(self, num_input):
    super(Reverse, self).__init__()
    self.perm = np.array(np.arange(0, num_input)[::-1])
    self.inv_perm = np.argsort(self.perm)

  def forward(self, inputs, context=None, mode='forward'):
    if mode == "forward":
      return inputs[:, :, self.perm], torch.zeros_like(inputs, device=inputs.device)
    elif mode == "inverse":
      return inputs[:, :, self.inv_perm], torch.zeros_like(inputs, device=inputs.device)
    else:
      raise ValueError("Mode must be one of {forward, inverse}.")

    

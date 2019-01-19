"""Fit a VAE to MNIST. 

Conventions:
  - batch size is the innermost dimension, then the sample dimension, then latent dimension
"""
import torch
import torch.utils
from torch import nn
import nomen
import yaml
import numpy as np
import logging

import data

config = """
latent_size: 128
data_size: 784
learning_rate: 0.001
batch_size: 128
test_batch_size: 512
max_iterations: 100000
log_interval: 1000
n_samples: 77
"""

class NeuralNetwork(nn.Module):
  def __init__(self, input_size, output_size, hidden_size):
    super().__init__()
    modules = [nn.Linear(input_size, hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size, hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size, output_size)]
    self.net = nn.Sequential(*modules)
    
  def forward(self, input):
    return self.net(input)



class Model(nn.Module):
  """Bernoulli model parameterized by a generative network with Gaussian latents for MNIST."""
  def __init__(self, latent_size, data_size, batch_size):
    super().__init__()
    # prior on latents is standard normal
    self.p_z = torch.distributions.Normal(torch.zeros(latent_size), torch.ones(latent_size))
    # likelihood is bernoulli, equivalent to negative binary cross entropy
    self.log_p_x = BernoulliLogProb()
    # generative network is a MLP
    self.generative_network = NeuralNetwork(input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2)
    

  def forward(self, z, x):
    """Return log probability of model."""
    log_p_z = self.p_z.log_prob(z).sum(-1)
    logits = self.generative_network(z)
    log_p_x = self.log_p_x(logits, x).sum(-1)
    return log_p_z + log_p_x


class NormalLogProb(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, loc, scale, z):
    var = torch.pow(scale, 2)
    return -0.5 * torch.log(2 * np.pi * var) + torch.pow(z - loc, 2) / (2 * var)

class BernoulliLogProb(nn.Module):
  def __init__(self):
    super().__init__()
    self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, logits, target):
    logits, target = torch.broadcast_tensors(logits, target.unsqueeze(1))
    return -self.bce_with_logits(logits, target)
    
class Variational(nn.Module):
  """Approximate posterior parameterized by an inference network."""
  def __init__(self, latent_size, data_size):
    super().__init__()
    self.inference_network = NeuralNetwork(input_size=data_size, output_size=latent_size * 2, hidden_size=latent_size*2)
    self.log_q_z = NormalLogProb()
    self.softplus = nn.Softplus()

  def forward(self, x, n_samples=1):
    """Return sample of latent variable and log prob."""
    loc, scale_arg = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=2, dim=-1)
    scale = self.softplus(scale_arg)
    eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]))
    z = loc + scale * eps  # reparameterization
    log_q_z = self.log_q_z(loc, scale, z).sum(-1)
    return z, log_q_z


def cycle(iterable):
  while True:
    for x in iterable:
      yield x


def evaluate(n_samples, model, variational, eval_data):
  model.eval()
  total_log_p_x = 0.0
  total_elbo = 0.0
  for batch in eval_data:
    x = batch[0]
    z, log_q_z = variational(x, n_samples)
    log_p_x_and_z = model(z, x)
    # importance sampling of approximate marginal likelihood
    # using logsumexp in the sample dimension
    elbo = log_p_x_and_z - log_q_z
    log_p_x = torch.logsumexp(elbo, dim=1) - np.log(n_samples)
    # average over sample dimension, sum over minibatch
    total_elbo += elbo.cpu().numpy().mean(1).sum()
    # sum over minibatch
    total_log_p_x += log_p_x.cpu().numpy().sum()
  n_data = len(eval_data.dataset)
  return total_elbo / n_data, total_log_p_x / n_data
  
  
if __name__ == '__main__':
  dictionary = yaml.load(config)
  cfg = nomen.Config(dictionary)
  
  model = Model(latent_size=cfg.latent_size, data_size=cfg.data_size, batch_size=cfg.batch_size)
  variational = Variational(latent_size=cfg.latent_size, data_size=cfg.data_size)

  optimizer = torch.optim.RMSprop(list(model.parameters()) + list(variational.parameters()), 
                                  lr=cfg.learning_rate)

  train_data, valid_data, test_data = data.load_binary_mnist(cfg)

  for step, batch in enumerate(cycle(train_data)):
    x = batch[0]
    model.zero_grad()
    variational.zero_grad()
    z, log_q_z = variational(x)
    log_p_x_and_z = model(z, x)
    elbo = (log_p_x_and_z - log_q_z).mean(1)
    loss = -elbo.mean(0)
    loss.backward()
    optimizer.step()

    if step % cfg.log_interval == 0:
      print(f'step:\t{step}\ttrain elbo: {elbo.detach().cpu().numpy()[0]:.2f}')
      with torch.no_grad():
        valid_elbo, valid_log_p_x = evaluate(cfg.n_samples, model, variational, valid_data)
      print(f'step:\t{step}\tvalid elbo: {valid_elbo:.2f}\tvalid log p(x): {valid_log_p_x:.2f}')

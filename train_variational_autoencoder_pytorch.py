"""Fit a variational autoencoder to MNIST. 

Notes:
  - run https://github.com/altosaar/proximity_vi/blob/master/get_binary_mnist.py to download binary MNIST file
  - batch size is the innermost dimension, then the sample dimension, then latent dimension
"""
import torch
import torch.utils
import torch.utils.data
from torch import nn
import nomen
import yaml
import numpy as np
import logging
import pathlib
import h5py
import random
import data
import flow

config = """
latent_size: 128
variational: flow
flow_depth: 2
data_size: 784
learning_rate: 0.001
batch_size: 128
test_batch_size: 512
max_iterations: 100000
log_interval: 10000
early_stopping_interval: 5
n_samples: 128
use_gpu: true
train_dir: $TMPDIR
data_dir: $TMPDIR
seed: 582838
"""

class Model(nn.Module):
  """Bernoulli model parameterized by a generative network with Gaussian latents for MNIST."""
  def __init__(self, latent_size, data_size):
    super().__init__()
    self.register_buffer('p_z_loc', torch.zeros(latent_size))
    self.register_buffer('p_z_scale', torch.ones(latent_size))
    self.log_p_z = NormalLogProb()
    self.log_p_x = BernoulliLogProb()
    self.generative_network = NeuralNetwork(input_size=latent_size,
                                            output_size=data_size, 
                                            hidden_size=latent_size * 2)

  def forward(self, z, x):
    """Return log probability of model."""
    log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
    logits = self.generative_network(z)
    # unsqueeze sample dimension
    logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
    log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
    return log_p_z + log_p_x

    
class VariationalMeanField(nn.Module):
  """Approximate posterior parameterized by an inference network."""
  def __init__(self, latent_size, data_size):
    super().__init__()
    self.inference_network = NeuralNetwork(input_size=data_size, 
                                           output_size=latent_size * 2, 
                                           hidden_size=latent_size*2)
    self.log_q_z = NormalLogProb()
    self.softplus = nn.Softplus()

  def forward(self, x, n_samples=1):
    """Return sample of latent variable and log prob."""
    loc, scale_arg = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=2, dim=-1)
    scale = self.softplus(scale_arg)
    eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
    z = loc + scale * eps  # reparameterization
    log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
    return z, log_q_z


class VariationalFlow(nn.Module):
  """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""
  def __init__(self, latent_size, data_size, flow_depth):
    super().__init__()
    hidden_size = latent_size * 2
    self.inference_network = NeuralNetwork(input_size=data_size, 
                                           # loc, scale, and context
                                           output_size=latent_size * 3, 
                                           hidden_size=hidden_size)
    modules = []
    for _ in range(flow_depth):
      modules.append(flow.InverseAutoregressiveFlow(num_input=latent_size,
                                                    num_hidden=hidden_size,
                                                    num_context=latent_size))
      modules.append(flow.Reverse(latent_size))
    self.q_z_flow = flow.FlowSequential(*modules)
    self.log_q_z_0 = NormalLogProb()
    self.softplus = nn.Softplus()

  def forward(self, x, n_samples=1):
    """Return sample of latent variable and log prob."""
    loc, scale_arg, h = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=3, dim=-1)
    scale = self.softplus(scale_arg)
    eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
    z_0 = loc + scale * eps  # reparameterization
    log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
    z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
    log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
    return z_T, log_q_z



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


class NormalLogProb(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, loc, scale, z):
    var = torch.pow(scale, 2)
    return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
  def __init__(self):
    super().__init__()
    self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, logits, target):
    # bernoulli log prob is equivalent to negative binary cross entropy
    return -self.bce_with_logits(logits, target)


def cycle(iterable):
  while True:
    for x in iterable:
      yield x


def load_binary_mnist(cfg, **kwcfg):
  fname = cfg.data_dir / 'binary_mnist.h5'
  if not fname.exists():
    print('Downloading binary MNIST data...')
    data.download_binary_mnist(fname)
  f = h5py.File(pathlib.os.path.join(pathlib.os.environ['DAT'], 'binary_mnist.h5'), 'r')
  x_train = f['train'][::]
  x_val = f['valid'][::]
  x_test = f['test'][::]
  train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
  train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True, **kwcfg)
  validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
  val_loader = torch.utils.data.DataLoader(validation, batch_size=cfg.test_batch_size, shuffle=False)
  test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
  test_loader = torch.utils.data.DataLoader(test, batch_size=cfg.test_batch_size, shuffle=False)
  return train_loader, val_loader, test_loader


def evaluate(n_samples, model, variational, eval_data):
  model.eval()
  total_log_p_x = 0.0
  total_elbo = 0.0
  for batch in eval_data:
    x = batch[0].to(next(model.parameters()).device)
    z, log_q_z = variational(x, n_samples)
    log_p_x_and_z = model(z, x)
    # importance sampling of approximate marginal likelihood with q(z)
    # as the proposal, and logsumexp in the sample dimension
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
  cfg.parse_args()
  device = torch.device("cuda:0" if cfg.use_gpu else "cpu")
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  random.seed(cfg.seed)

  model = Model(latent_size=cfg.latent_size, 
                data_size=cfg.data_size)
  if cfg.variational == 'flow':
    variational = VariationalFlow(latent_size=cfg.latent_size,
                                  data_size=cfg.data_size,
                                  flow_depth=cfg.flow_depth)
  elif cfg.variational == 'mean-field':
    variational = VariationalMeanField(latent_size=cfg.latent_size,
                                       data_size=cfg.data_size)
  else:
    raise ValueError('Variational distribution not implemented: %s' % cfg.variational)

  model.to(device)
  variational.to(device)

  optimizer = torch.optim.RMSprop(list(model.parameters()) +
                                  list(variational.parameters()),
                                  lr=cfg.learning_rate,
                                  centered=True)

  kwargs = {'num_workers': 4, 'pin_memory': True} if cfg.use_gpu else {}
  train_data, valid_data, test_data = load_binary_mnist(cfg, **kwargs)

  best_valid_elbo = -np.inf
  num_no_improvement = 0

  for step, batch in enumerate(cycle(train_data)):
    x = batch[0].to(device)
    model.zero_grad()
    variational.zero_grad()
    z, log_q_z = variational(x, n_samples=1)
    log_p_x_and_z = model(z, x)
    # average over sample dimension
    elbo = (log_p_x_and_z - log_q_z).mean(1)
    # sum over batch dimension
    loss = -elbo.sum(0)
    loss.backward()
    optimizer.step()

    if step % cfg.log_interval == 0:
      print(f'step:\t{step}\ttrain elbo: {elbo.detach().cpu().numpy().mean():.2f}')
      with torch.no_grad():
        valid_elbo, valid_log_p_x = evaluate(cfg.n_samples, model, variational, valid_data)
      print(f'step:\t{step}\t\tvalid elbo: {valid_elbo:.2f}\tvalid log p(x): {valid_log_p_x:.2f}')
      if valid_elbo > best_valid_elbo:
        num_no_improvement = 0
        best_valid_elbo = valid_elbo
        states = {'model': model.state_dict(),
                  'variational': variational.state_dict()}
        torch.save(states, cfg.train_dir / 'best_state_dict')
      else:
        num_no_improvement += 1

      if num_no_improvement > cfg.early_stopping_interval:
        checkpoint = torch.load(cfg.train_dir / 'best_state_dict')
        model.load_state_dict(checkpoint['model'])
        variational.load_state_dict(checkpoint['variational'])
        with torch.no_grad():
          test_elbo, test_log_p_x = evaluate(cfg.n_samples, model, variational, test_data)
        print(f'step:\t{step}\t\ttest elbo: {test_elbo:.2f}\ttest log p(x): {test_log_p_x:.2f}')
        break

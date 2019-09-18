import itertools
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.contrib.slim as slim
import time
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from imageio import imwrite
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tfkl = tfk.layers
tfc = tf.compat.v1

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/dat/', 'Directory for data')
flags.DEFINE_string('logdir', '/tmp/log/', 'Directory for logs')
flags.DEFINE_integer('latent_dim', 100, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 64, 'Minibatch size')
flags.DEFINE_integer('n_samples', 1, 'Number of samples to save')
flags.DEFINE_integer('print_every', 1000, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 200, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 100000, 'number of iterations')

FLAGS = flags.FLAGS


def inference_network(x, latent_dim, hidden_size):
  """Construct an inference network parametrizing a Gaussian.

  Args:
    x: A batch of MNIST digits.
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.

  Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
  """
  inference_net = tfk.Sequential([
    tfkl.Flatten(),
    tfkl.Dense(hidden_size, activation=tf.nn.relu),
    tfkl.Dense(hidden_size, activation=tf.nn.relu),
    tfkl.Dense(latent_dim * 2, activation=None)
    ])
  gaussian_params = inference_net(x)
  # The mean parameter is unconstrained
  mu = gaussian_params[:, :latent_dim]
  # The standard deviation must be positive. Parametrize with a softplus
  sigma = tf.nn.softplus(gaussian_params[:, latent_dim:])
  return mu, sigma


def generative_network(z, hidden_size):
  """Build a generative network parametrizing the likelihood of the data

  Args:
    z: Samples of latent variables
    hidden_size: Size of the hidden state of the neural net

  Returns:
    bernoulli_logits: logits for the Bernoulli likelihood of the data
  """
  generative_net = tfk.Sequential([
    tfkl.Dense(hidden_size, activation=tf.nn.relu),
    tfkl.Dense(hidden_size, activation=tf.nn.relu),
    tfkl.Dense(28 * 28, activation=None)
    ])
  bernoulli_logits = generative_net(z)
  return tf.reshape(bernoulli_logits, [-1, 28, 28, 1])


def train():
  # Train a Variational Autoencoder on MNIST

  # Input placeholders
  with tf.name_scope('data'):
    x = tfc.placeholder(tf.float32, [None, 28, 28, 1])
    tfc.summary.image('data', x)

  with tfc.variable_scope('variational'):
    q_mu, q_sigma = inference_network(x=x,
                                      latent_dim=FLAGS.latent_dim,
                                      hidden_size=FLAGS.hidden_size)
    # The variational distribution is a Normal with mean and standard
    # deviation given by the inference network
    q_z = tfp.distributions.Normal(loc=q_mu, scale=q_sigma)
    assert q_z.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED

  with tfc.variable_scope('model'):
    # The likelihood is Bernoulli-distributed with logits given by the
    # generative network
    p_x_given_z_logits = generative_network(z=q_z.sample(),
                                            hidden_size=FLAGS.hidden_size)
    p_x_given_z = tfp.distributions.Bernoulli(logits=p_x_given_z_logits)
    posterior_predictive_samples = p_x_given_z.sample()
    tfc.summary.image('posterior_predictive',
                     tf.cast(posterior_predictive_samples, tf.float32))

  # Take samples from the prior
  with tfc.variable_scope('model', reuse=True):
    p_z = tfp.distributions.Normal(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                               scale=np.ones(FLAGS.latent_dim, dtype=np.float32))
    p_z_sample = p_z.sample(FLAGS.n_samples)
    p_x_given_z_logits = generative_network(z=p_z_sample,
                                            hidden_size=FLAGS.hidden_size)
    prior_predictive = tfp.distributions.Bernoulli(logits=p_x_given_z_logits)
    prior_predictive_samples = prior_predictive.sample()
    tfc.summary.image('prior_predictive',
                     tf.cast(prior_predictive_samples, tf.float32))

  # Take samples from the prior with a placeholder
  with tfc.variable_scope('model', reuse=True):
    z_input = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
    p_x_given_z_logits = generative_network(z=z_input,
                                            hidden_size=FLAGS.hidden_size)
    prior_predictive_inp = tfp.distributions.Bernoulli(logits=p_x_given_z_logits)
    prior_predictive_inp_sample = prior_predictive_inp.sample()

  # Build the evidence lower bound (ELBO) or the negative loss
  kl = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
  expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x),
                                          [1, 2, 3])

  elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)
  optimizer = tfc.train.RMSPropOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(-elbo)

  # Merge all the summaries
  summary_op = tfc.summary.merge_all()

  init_op = tfc.global_variables_initializer()

  # Run training
  sess = tfc.InteractiveSession()
  sess.run(init_op)

  mnist_data = tfds.load(name='binarized_mnist', split='train', shuffle_files=False)
  dataset = mnist_data.repeat().shuffle(buffer_size=1024).batch(FLAGS.batch_size)

  print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)
  train_writer = tfc.summary.FileWriter(FLAGS.logdir, sess.graph)

  t0 = time.time()
  for i, batch in enumerate(tfds.as_numpy(dataset)):
    np_x = batch['image']
    sess.run(train_op, {x: np_x})
    if i % FLAGS.print_every == 0:
      np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
      train_writer.add_summary(summary_str, i)
      print('Iteration: {0:d} ELBO: {1:.3f} s/iter: {2:.3e}'.format(
          i,
          np_elbo / FLAGS.batch_size,
          (time.time() - t0) / FLAGS.print_every))
      # Save samples
      np_posterior_samples, np_prior_samples = sess.run(
          [posterior_predictive_samples, prior_predictive_samples], {x: np_x})
      for k in range(FLAGS.n_samples):
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_data.jpg' % (i, k))
        imwrite(f_name, np_x[k, :, :, 0].astype(np.uint8))
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_sample.jpg' % (i, k))
        imwrite(f_name, np_posterior_samples[k, :, :, 0].astype(np.uint8))
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_prior_predictive_%d.jpg' % (i, k))
        imwrite(f_name, np_prior_samples[k, :, :, 0].astype(np.uint8))
      t0 = time.time()

if __name__ == '__main__':
  train()

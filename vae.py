import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from scipy.misc import imsave
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_string('logdir', '/tmp/logs/', 'Directory for storing data')
flags.DEFINE_integer('latent_dim', 100, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 64, 'Minibatch size')
flags.DEFINE_integer('n_samples', 10, 'Number of samples to save')
flags.DEFINE_integer('print_every', 1000, 'Print every n iterations')

FLAGS = flags.FLAGS


def inference_network(x, latent_dim, hidden_size):
  """Construct an inference network parametrizing a Gaussian.

  Args:
    x: A batch of MNIST digits.
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.
  """
  with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
    net = slim.flatten(x)
    net = slim.fully_connected(net, hidden_size)
    net = slim.fully_connected(net, hidden_size)
    gaussian_params = slim.fully_connected(
        net, latent_dim * 2, activation_fn=None)
  mu = gaussian_params[:, :latent_dim]
  sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, latent_dim:])
  return mu, sigma


def generative_network(z, hidden_size):
  with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
    net = slim.fully_connected(z, hidden_size)
    net = slim.fully_connected(net, hidden_size)
    bernoulli_logits = slim.fully_connected(net, 784, activation_fn=None)
    bernoulli_logits = tf.reshape(bernoulli_logits, [-1, 28, 28, 1])
  return bernoulli_logits


def train():
  # Train a Variational Autoencoder on MNIST

  # Input placehoolders
  with tf.name_scope('data'):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tf.image_summary('data', x, max_images=10)


  with tf.variable_scope('variational'):
    q_mu, q_sigma = inference_network(x=x,
                                      latent_dim=FLAGS.latent_dim,
                                      hidden_size=200)
    with sg.value_type(sg.SampleAndReshapeValue()):
      q_z = sg.DistributionTensor(distributions.Normal, mu=q_mu, sigma=q_sigma)

  with tf.variable_scope('model'):
    p_x_given_z_logits = generative_network(z=q_z, hidden_size=200)
    posterior_predictive = distributions.Bernoulli(logits=p_x_given_z_logits)
    posterior_predictive_samples = posterior_predictive.sample()
    tf.image_summary('posterior_predictive',
                     tf.cast(posterior_predictive_samples, tf.float32),
                     max_images=10)


  with tf.variable_scope('model', reuse=True):
    p_z = distributions.Normal(mu=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                               sigma=np.ones(FLAGS.latent_dim, dtype=np.float32))
    p_z_sample = p_z.sample_n(FLAGS.n_samples)
    p_x_given_z_logits = generative_network(z=p_z_sample, hidden_size=200)
    prior_predictive = distributions.Bernoulli(logits=p_x_given_z_logits)
    prior_predictive_samples = prior_predictive.sample()
    tf.image_summary('prior_predictive',
                     tf.cast(prior_predictive_samples, tf.float32),
                     max_images=10)


  # Build the evidence lower bound (ELBO) or the negative loss
  kl = tf.reduce_sum(distributions.kl(q_z.distribution, p_z), 1)
  expected_log_likelihood = tf.reduce_sum(posterior_predictive.log_pmf(x),
                                          [1, 2, 3])

  elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)

  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

  train_op = optimizer.minimize(-elbo)

  # Merge all the summaries
  summary_op = tf.merge_all_summaries()

  init_op = tf.initialize_all_variables()

  # Run training
  sess = tf.InteractiveSession()
  sess.run(init_op)

  mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

  print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)
  train_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)

  for i in range(100000):
    np_x, _ = mnist.train.next_batch(FLAGS.batch_size)
    np_x = np_x.reshape(FLAGS.batch_size, 28, 28, 1)
    np_x = (np_x > 0.5).astype(np.float32)
    sess.run(train_op, feed_dict={x: np_x})

    # Print progress and save samples every so often
    t0 = time.time()
    if i % FLAGS.print_every == 0:
      np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
      train_writer.add_summary(summary_str, i)
      print('Iteration: {0:d} ELBO: {1:.3f} Examples/s: {2:.3e}'.format(
          i,
          np_elbo / FLAGS.batch_size,
          FLAGS.batch_size * FLAGS.print_every / (time.time() - t0)))
      t0 = time.time()

      # Save samples
      np_posterior_samples, np_prior_samples = sess.run(
          [posterior_predictive_samples, prior_predictive_samples], {x: np_x})
      for k in range(FLAGS.n_samples):
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_data.jpg' % (i, k))
        imsave(f_name, np_x[k, :, :, 0])
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_sample.jpg' % (i, k))
        imsave(f_name, np_posterior_samples[k, :, :, 0])
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_prior_predictive_%d.jpg' % (i, k))
        imsave(f_name, np_prior_samples[k, :, :, 0])


def main(_):
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  train()

if __name__ == '__main__':
  tf.app.run()

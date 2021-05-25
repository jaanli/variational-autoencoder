"""Train variational autoencoder or binary MNIST data.

Largely follows https://github.com/deepmind/dm-haiku/blob/master/examples/vae.py"""

import argparse
import pathlib
from calendar import c
from typing import Generator, Mapping, NamedTuple, Sequence, Tuple

import jax
import numpy as np

jax.config.update("jax_platform_name", "cpu")  # suppress warning about no GPUs

import haiku as hk
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Batch = Mapping[str, np.ndarray]
MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)
PRNGKey = jnp.ndarray


def add_args(parser):
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--variational", choices=["flow", "mean-field"])
    parser.add_argument("--flow_depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--training_steps", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--early_stopping_interval", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument(
        "--use_gpu", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_dir", type=pathlib.Path, default="/tmp")
    parser.add_argument("--data_dir", type=pathlib.Path, default="/tmp")


def load_dataset(
    split: str, batch_size: int, seed: int, repeat: bool = False
) -> Generator[Batch, None, None]:
    ds = tfds.load(
        "binarized_mnist",
        split=split,
        shuffle_files=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed),
    )
    ds = ds.shuffle(buffer_size=10 * batch_size, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=5)
    if repeat:
        ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


class Model(hk.Module):
    """Deep latent Gaussian model or variational autoencoder."""

    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._output_shape = output_shape
        self.generative_network = hk.Sequential(
            [
                hk.Linear(self._hidden_size),
                jax.nn.relu,
                # hk.Linear(self._hidden_size),
                # jax.nn.relu,
                hk.Linear(np.prod(self._output_shape)),
                hk.Reshape(self._output_shape, preserve_dims=2),
            ]
        )

    def __call__(self, x: jnp.ndarray, z: jnp.ndarray) -> Tuple[tfd.Distribution]:
        p_z = tfd.Normal(
            loc=jnp.zeros(self._latent_size), scale=jnp.ones(self._latent_size)
        )
        logits = self.generative_network(z)
        p_x_given_z = tfd.Bernoulli(logits=logits)
        return p_z, p_x_given_z


class VariationalMeanField(hk.Module):
    """Mean field variational distribution q(z | x) parameterized by inference network."""

    def __init__(self, latent_size: int, hidden_size: int):
        super().__init__()
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self.inference_network = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(self._hidden_size),
                jax.nn.relu,
                # hk.Linear(self._hidden_size),
                # jax.nn.relu,
                hk.Linear(self._latent_size * 2),
            ]
        )

    def condition(self, inputs):
        """Compute parameters of a multivariate independent Normal distribution based on the inputs."""
        out = self.inference_network(inputs)
        loc, scale_arg = jnp.split(out, 2, axis=-1)
        scale = jax.nn.softplus(scale_arg)
        return loc, scale

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        loc, scale = self.condition(x)
        # IMPORTANT: need to check in source code that reparameterization_type=tfd.FULLY_REPARAMETERIZED for this class
        q_z = tfd.Normal(loc=loc, scale=scale)
        return q_z


class ModelAndVariationalOutput(NamedTuple):
    p_z: tfd.Distribution
    p_x_given_z: tfd.Distribution
    q_z: tfd.Distribution
    z: jnp.ndarray


class ModelAndVariational(hk.Module):
    """Parent class for creating inputs to the variational inference algorithm."""

    def __init__(self, latent_size: int, hidden_size: int, output_shape: Sequence[int]):
        super().__init__()
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, x: jnp.ndarray) -> ModelAndVariationalOutput:
        x = x.astype(jnp.float32)
        q_z = VariationalMeanField(self._latent_size, self._hidden_size)(x)
        # use a single sample from variational distribution to train
        # shape [num_samples, batch_size, latent_size]
        z = q_z.sample(sample_shape=[1], seed=hk.next_rng_key())

        p_z, p_x_given_z = Model(
            self._latent_size, self._hidden_size, MNIST_IMAGE_SHAPE
        )(x=x, z=z)
        return ModelAndVariationalOutput(p_z, p_x_given_z, q_z, z)


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    model_and_variational = hk.transform(
        lambda x: ModelAndVariational(
            args.latent_size, args.hidden_size, MNIST_IMAGE_SHAPE
        )(x)
    )

    # @jax.jit
    def objective_fn(params: hk.Params, rng_key: PRNGKey, batch: Batch) -> jnp.ndarray:
        x = batch["image"]
        out: ModelAndVariationalOutput = model_and_variational.apply(params, rng_key, x)
        log_q_z = out.q_z.log_prob(out.z).sum(axis=-1)
        # sum over last three image dimensions (width, height, channels)
        log_p_x_given_z = out.p_x_given_z.log_prob(x).sum(axis=(-3, -2, -1))
        # sum over latent dimension
        log_p_z = out.p_z.log_prob(out.z).sum(axis=-1)

        elbo = log_p_x_given_z + log_p_z - log_q_z
        # average elbo over number of samples
        elbo = elbo.mean(axis=0)
        # sum elbo over batch
        elbo = elbo.sum(axis=0)
        return -elbo

    rng_seq = hk.PRNGSequence(args.random_seed)

    params = model_and_variational.init(
        next(rng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE))
    )
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    # @jax.jit
    def train_step(
        params: hk.Params, rng_key: PRNGKey, opt_state: optax.OptState, batch: Batch
    ) -> Tuple[hk.Params, optax.OptState]:
        """Single update step to maximize the ELBO."""
        grads = jax.grad(objective_fn)(params, rng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    @jax.jit
    def importance_weighted_estimate(
        params: hk.Params, rng_key: PRNGKey, batch: Batch
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Estimate marginal log p(x) using importance sampling."""
        x = batch["image"]
        out: ModelAndVariationalOutput = model_and_variational.apply(params, rng_key, x)
        log_q_z = out.q_z.log_prob(out.z).sum(axis=-1)
        # sum over last three image dimensions (width, height, channels)
        log_p_x_given_z = out.p_x_given_z.log_prob(x).sum(axis=(-3, -2, -1))
        # sum over latent dimension
        log_p_z = out.p_z.log_prob(out.z).sum(axis=-1)

        elbo = log_p_x_given_z + log_p_z - log_q_z
        # importance sampling of approximate marginal likelihood with q(z)
        # as the proposal, and logsumexp in the sample dimension
        log_p_x = jax.nn.logsumexp(elbo, axis=0) - jnp.log(jnp.shape(elbo)[0])
        # sum over the elements of the minibatch
        log_p_x = log_p_x.sum(0)
        # average elbo over number of samples
        elbo = elbo.mean(axis=0)
        # sum elbo over batch
        elbo = elbo.sum(axis=0)
        return elbo, log_p_x

    def evaluate(
        dataset: Generator[Batch, None, None],
        params: hk.Params,
        rng_seq: hk.PRNGSequence,
    ) -> Tuple[float, float]:
        total_elbo = 0.0
        total_log_p_x = 0.0
        dataset_size = 0
        for batch in dataset:
            elbo, log_p_x = importance_weighted_estimate(params, next(rng_seq), batch)
            total_elbo += elbo
            total_log_p_x += log_p_x
            dataset_size += len(batch["image"])
        return total_elbo / dataset_size, total_log_p_x / dataset_size

    train_ds = load_dataset(
        tfds.Split.TRAIN, args.batch_size, args.random_seed, repeat=True
    )
    test_ds = load_dataset(tfds.Split.TEST, args.batch_size, args.random_seed)

    for step in range(args.training_steps):
        params, opt_state = train_step(params, next(rng_seq), opt_state, next(train_ds))
        if step % args.log_interval == 0:
            valid_ds = load_dataset(
                tfds.Split.VALIDATION, args.batch_size, args.random_seed
            )
            elbo, log_p_x = evaluate(valid_ds, params, rng_seq)
            train_elbo = (
                -objective_fn(params, next(rng_seq), next(train_ds)) / args.batch_size
            )
            print(
                f"Step {step:<10d}\t"
                f"Train ELBO estimate: {train_elbo:<5.3f}\t"
                f"Validation ELBO estimate: {elbo:<5.3f}\t"
                f"Validation log p(x) estimate: {log_p_x:<5.3f}"
            )


if __name__ == "__main__":
    main()


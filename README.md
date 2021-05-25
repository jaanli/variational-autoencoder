# Variational Autoencoder in tensorflow and pytorch
[![DOI](https://zenodo.org/badge/65744394.svg)](https://zenodo.org/badge/latestdoi/65744394)

Reference implementation for a variational autoencoder in TensorFlow and PyTorch.

I recommend the PyTorch version. It includes an example of a more expressive variational family, the [inverse autoregressive flow](https://arxiv.org/abs/1606.04934).

Variational inference is used to fit the model to binarized MNIST handwritten digits images. An inference network (encoder) is used to amortize the inference and share parameters across datapoints. The likelihood is parameterized by a generative network (decoder).

Blog post: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

Example output with importance sampling for estimating the marginal likelihood on Hugo Larochelle's Binary MNIST dataset. Final marginal likelihood on the test set was `-97.10` nats after 65k iterations.

```
$ python train_variational_autoencoder_pytorch.py --variational mean-field
step:   0       train elbo: -558.28
step:   0               valid elbo: -392.78     valid log p(x): -359.91
step:   10000   train elbo: -106.67
step:   10000           valid elbo: -109.12     valid log p(x): -103.11
step:   20000   train elbo: -107.28
step:   20000           valid elbo: -105.65     valid log p(x): -99.74
```


Using a non mean-field, more expressive variational posterior approximation (inverse autoregressive flow, https://arxiv.org/abs/1606.04934), the test marginal log-likelihood improves to `-95.33` nats:

```
$ python train_variational_autoencoder_pytorch.py --variational flow
step:   0       train elbo: -578.35
step:   0               valid elbo: -407.06     valid log p(x): -367.88
step:   10000   train elbo: -106.63
step:   10000           valid elbo: -110.12     valid log p(x): -104.00
step:   20000   train elbo: -101.51
step:   20000           valid elbo: -105.02     valid log p(x): -99.11
step:   30000   train elbo: -98.70
step:   30000           valid elbo: -103.76     valid log p(x): -97.71
```

Using jax (anaconda environment is in `environment-jax.yml`):
```
Step 0          Train ELBO estimate: -565.785   Validation ELBO estimate: -565.775      Validation log p(x) estimate: -565.775  Speed: 3813003636 examples/s
Step 10000      Train ELBO estimate: -99.048    Validation ELBO estimate: -105.412      Validation log p(x) estimate: -105.412  Speed: 134   examples/s
Step 20000      Train ELBO estimate: -108.399   Validation ELBO estimate: -105.191      Validation log p(x) estimate: -105.191  Speed: 140   examples/s
Step 30000      Train ELBO estimate: -100.839   Validation ELBO estimate: -105.404      Validation log p(x) estimate: -105.404  Speed: 139   examples/s
Step 40000      Train ELBO estimate: -97.761    Validation ELBO estimate: -105.382      Validation log p(x) estimate: -105.382  Speed: 139   examples/s
Step 50000      Train ELBO estimate: -98.228    Validation ELBO estimate: -105.718      Validation log p(x) estimate: -105.718  Speed: 139   examples/s
```
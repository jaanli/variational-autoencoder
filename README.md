# Variational Autoencoder in tensorflow and pytorch
[![DOI](https://zenodo.org/badge/65744394.svg)](https://zenodo.org/badge/latestdoi/65744394)

Reference implementation for a variational autoencoder in TensorFlow and PyTorch.

I recommend the PyTorch version. It includes an example of a more expressive variational family, the [inverse autoregressive flow](https://arxiv.org/abs/1606.04934).

Variational inference is used to fit the model to binarized MNIST handwritten digits images. An inference network (encoder) is used to amortize the inference and share parameters across datapoints. The likelihood is parameterized by a generative network (decoder).

Blog post: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/


## PyTorch implementation

(anaconda environment is in `environment-jax.yml`)

Importance sampling is used to estimate the marginal likelihood on Hugo Larochelle's Binary MNIST dataset. The final marginal likelihood on the test set was `-97.10` nats is comparable to published numbers.

```
$ python train_variational_autoencoder_pytorch.py --variational mean-field --use_gpu --data_dir $DAT --max_iterations 30000 --log_interval 10000
Step 0          Train ELBO estimate: -558.027   Validation ELBO estimate: -384.432      Validation log p(x) estimate: -355.430  Speed: 2.72e+06 examples/s
Step 10000      Train ELBO estimate: -111.323   Validation ELBO estimate: -109.048      Validation log p(x) estimate: -103.746  Speed: 2.64e+04 examples/s
Step 20000      Train ELBO estimate: -103.013   Validation ELBO estimate: -107.655      Validation log p(x) estimate: -101.275  Speed: 2.63e+04 examples/s
Step 29999      Test ELBO estimate: -106.642    Test log p(x) estimate: -100.309
Total time: 2.49 minutes
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

## jax implementation

Using jax (anaconda environment is in `environment-jax.yml`), to get a 3x speedup over pytorch:
```
$ python train_variational_autoencoder_jax.py --variational mean-field 
Step 0          Train ELBO estimate: -566.059   Validation ELBO estimate: -565.755      Validation log p(x) estimate: -557.914  Speed: 2.56e+11 examples/s
Step 10000      Train ELBO estimate: -98.560    Validation ELBO estimate: -105.725      Validation log p(x) estimate: -98.973   Speed: 7.03e+04 examples/s
Step 20000      Train ELBO estimate: -109.794   Validation ELBO estimate: -105.756      Validation log p(x) estimate: -97.914   Speed: 4.26e+04 examples/s
Step 29999      Test ELBO estimate: -104.867    Test log p(x) estimate: -96.716
Total time: 0.810 minutes
```

Inverse autoregressive flow in jax:
```
$ python train_variational_autoencoder_jax.py --variational flow 
Step 0          Train ELBO estimate: -727.404   Validation ELBO estimate: -726.977      Validation log p(x) estimate: -713.389  Speed: 2.56e+11 examples/s
Step 10000      Train ELBO estimate: -100.093   Validation ELBO estimate: -106.985      Validation log p(x) estimate: -99.565   Speed: 2.57e+04 examples/s
Step 20000      Train ELBO estimate: -113.073   Validation ELBO estimate: -108.057      Validation log p(x) estimate: -98.841   Speed: 3.37e+04 examples/s
Step 29999      Test ELBO estimate: -106.803    Test log p(x) estimate: -97.620
Total time: 2.350 minutes
```

(The difference between a mean field and inverse autoregressive flow may be due to several factors, chief being the lack of convolutions in the implementation. Residual blocks are used in https://arxiv.org/pdf/1606.04934.pdf to get the ELBO closer to -80 nats.)

# Generating the GIFs

1. Run `python train_variational_autoencoder_tensorflow.py`
2. Install imagemagick (homebrew for Mac: https://formulae.brew.sh/formula/imagemagick or Chocolatey in Windows: https://community.chocolatey.org/packages/imagemagick.app)
3. Go to the directory where the jpg files are saved, and run the imagemagick command to generate the .gif: `convert -delay 20 -loop 0 *.jpg latent-space.gif`

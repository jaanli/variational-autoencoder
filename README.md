# Variational Autoencoder / Deep Latent Gaussian Model in tensorflow and pytorch
Reference implementation for a variational autoencoder in TensorFlow and PyTorch.

I recommend the PyTorch version. It includes an example of a more expressive variational family, the [inverse autoregressive flow](https://arxiv.org/abs/1606.04934).

Variational inference is used to fit the model to binarized MNIST handwritten digits images. An inference network (encoder) is used to amortize the inference and share parameters across datapoints. The likelihood is parameterized by a generative network (decoder).

Blog post: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

Example output with importance sampling for estimating the marginal likelihood on Hugo Larochelle's Binary MNIST dataset. Finaly marginal likelihood on the test set of `-97.10` nats.

```
$ python train_variational_autoencoder_pytorch.py --variational mean-field
step:   0       train elbo: -558.69
step:   0               valid elbo: -391.84     valid log p(x): -363.25
step:   5000    train elbo: -116.09
step:   5000            valid elbo: -112.57     valid log p(x): -107.01
step:   10000   train elbo: -105.82
step:   10000           valid elbo: -108.49     valid log p(x): -102.62
step:   15000   train elbo: -106.78
step:   15000           valid elbo: -106.97     valid log p(x): -100.97
step:   20000   train elbo: -108.43
step:   20000           valid elbo: -106.23     valid log p(x): -100.04
step:   25000   train elbo: -99.68
step:   25000           valid elbo: -104.89     valid log p(x): -98.83
step:   30000   train elbo: -96.71
step:   30000           valid elbo: -104.50     valid log p(x): -98.34
step:   35000   train elbo: -98.64
step:   35000           valid elbo: -104.05     valid log p(x): -97.87
step:   40000   train elbo: -93.60
step:   40000           valid elbo: -104.10     valid log p(x): -97.68
step:   45000   train elbo: -96.45
step:   45000           valid elbo: -104.58     valid log p(x): -97.76
step:   50000   train elbo: -101.63
step:   50000           valid elbo: -104.72     valid log p(x): -97.81
step:   55000   train elbo: -106.78
step:   55000           valid elbo: -105.14     valid log p(x): -98.06
step:   60000   train elbo: -100.58
step:   60000           valid elbo: -104.13     valid log p(x): -97.30
step:   65000   train elbo: -96.19
step:   65000           valid elbo: -104.46     valid log p(x): -97.43
step:   65000           test elbo: -103.31      test log p(x): -97.10
```


Using a non mean-field, more expressive variational posterior approximation, the test marginal log-likelihood improves to `-95.33` nats:

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
step:   40000   train elbo: -104.31
step:   40000           valid elbo: -103.71     valid log p(x): -97.27
step:   50000   train elbo: -97.20
step:   50000           valid elbo: -102.97     valid log p(x): -96.60
step:   60000   train elbo: -97.50
step:   60000           valid elbo: -102.82     valid log p(x): -96.49
step:   70000   train elbo: -94.68
step:   70000           valid elbo: -102.63     valid log p(x): -96.22
step:   80000   train elbo: -92.86
step:   80000           valid elbo: -102.53     valid log p(x): -96.09
step:   90000   train elbo: -93.83
step:   90000           valid elbo: -102.33     valid log p(x): -96.00
step:   100000  train elbo: -93.91
step:   100000          valid elbo: -102.48     valid log p(x): -95.92
step:   110000  train elbo: -94.34
step:   110000          valid elbo: -102.81     valid log p(x): -96.09
step:   120000  train elbo: -88.63
step:   120000          valid elbo: -102.53     valid log p(x): -95.80
step:   130000  train elbo: -96.61
step:   130000          valid elbo: -103.56     valid log p(x): -96.26
step:   140000  train elbo: -94.92
step:   140000          valid elbo: -102.81     valid log p(x): -95.86
step:   150000  train elbo: -97.84
step:   150000          valid elbo: -103.06     valid log p(x): -95.92
step:   150000          test elbo: -101.64      test log p(x): -95.33
```

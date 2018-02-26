# Variational Autoencoder (Deep Latent Gaussian Model) in tf
Reference implementation for a variational autoencoder in TensorFlow.

Mean-field variational inference is used to fit the model to binarized MNIST handwritten digits images. An inference network (encoder) is used to amortize the inference and share parameters across datapoints. The likelihood is parameterized by a generative network (decoder).

Blog post: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

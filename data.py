"""Get the binarized MNIST dataset and convert to hdf5.
From https://github.com/yburda/iwae/blob/master/datasets.py
"""
import urllib.request
import os
import numpy as np
import h5py
import torch


def parse_binary_mnist(data_dir):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.join(data_dir, "binarized_mnist_train.amat")) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype("float32")
    with open(os.path.join(data_dir, "binarized_mnist_valid.amat")) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype("float32")
    with open(os.path.join(data_dir, "binarized_mnist_test.amat")) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype("float32")
    return train_data, validation_data, test_data


def download_binary_mnist(fname):
    data_dir = "/tmp/"
    subdatasets = ["train", "valid", "test"]
    for subdataset in subdatasets:
        filename = "binarized_mnist_{}.amat".format(subdataset)
        url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
            subdataset
        )
        local_filename = os.path.join(data_dir, filename)
        urllib.request.urlretrieve(url, local_filename)

    train, validation, test = parse_binary_mnist(data_dir)

    data_dict = {"train": train, "valid": validation, "test": test}
    f = h5py.File(fname, "w")
    f.create_dataset("train", data=data_dict["train"])
    f.create_dataset("valid", data=data_dict["valid"])
    f.create_dataset("test", data=data_dict["test"])
    f.close()
    print(f"Saved binary MNIST data to: {fname}")


def load_binary_mnist(fname, batch_size, test_batch_size, use_gpu):
    f = h5py.File(fname, "r")
    x_train = f["train"][::]
    x_val = f["valid"][::]
    x_test = f["test"][::]
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
    kwargs = {"num_workers": 4, "pin_memory": True} if use_gpu else {}
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, **kwargs
    )
    validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
    val_loader = torch.utils.data.DataLoader(
        validation, batch_size=test_batch_size, shuffle=False, **kwargs
    )
    test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=test_batch_size, shuffle=False, **kwargs
    )
    return train_loader, val_loader, test_loader

"""Load from /home/USER/data/mnist or elsewhere; download if missing."""
import gzip
import os
from urllib.request import urlretrieve
import numpy as np


def download_mnist(path):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if none existent. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a one hot encoding of the correct class.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'mnist')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path1):
        """Return images loaded locally."""
        with gzip.open(path1) as f:
            # First 16 bytes are magic_number, n_images, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 28, 28).astype('float32') / 255

    def _labels(path2):
        """Return labels loaded locally."""
        with gzip.open(path2) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)
        return integer_labels

        # def _one_hot(int_labels):
        #     """Return matrix whose rows are one hot encodings of integers."""
        #     n_rows = len(int_labels)
        #     n_cols = int_labels.max() + 1
        #     one_hot = np.zeros((n_rows, n_cols), dtype='int8')
        #     one_hot[np.arange(n_rows), int_labels] = 1
        #     return one_hot
        #
        # return _one_hot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels

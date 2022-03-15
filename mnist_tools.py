import os
import gzip
import struct
import tempfile
import shutil
from urllib.request import urlretrieve
from urllib.parse import urljoin
import numpy as np


mnist_info = {
    'train' : { 'labels' : 'train-labels-idx1-ubyte.gz',
                'images' : 'train-images-idx3-ubyte.gz' },
    'test'  : { 'labels' : 't10k-labels-idx1-ubyte.gz',
                'images' : 't10k-images-idx3-ubyte.gz' },
    'base_url' : 'http://yann.lecun.com/exdb/mnist/'
}


def download_mnist_file(fname, target_dir, force=False):
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin(mnist_info['base_url'], fname)
        urlretrieve(url, target_fname)


def parse_labels(fname, target_dir):
    with gzip.open(os.path.join(target_dir, fname), 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        dt = np.dtype(np.uint8).newbyteorder('>')
        labels = np.frombuffer(f.read(), dtype=dt)
        return labels


def parse_images(fname, target_dir):
    with gzip.open(os.path.join(target_dir, fname), 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        nrows, ncols = struct.unpack('>II', f.read(8))
        dt = np.dtype(np.uint8).newbyteorder('>')
        images = np.frombuffer(f.read(), dtype=dt)
        images = images.reshape((size, nrows, ncols))
        return images


class MNIST(object):
    def __init__(self, target_dir=None, clean_up=True, force=False):
        if target_dir is None:
            self.target_dir = tempfile.mkdtemp(prefix='mnist')
            self.clean_up = True
        else:
            os.makedirs(target_dir, exist_ok=True)
            self.target_dir = target_dir
            self.clean_up = clean_up

        self.force = force


    def __del__(self):
        if self.clean_up:
            shutil.rmtree(self.target_dir)


    def train_labels(self):
        download_mnist_file(mnist_info['train']['labels'],
                            self.target_dir, force=self.force)
        return parse_labels(mnist_info['train']['labels'], self.target_dir)


    def train_images(self):
        download_mnist_file(mnist_info['train']['images'],
                            self.target_dir, force=self.force)
        return parse_images(mnist_info['train']['images'], self.target_dir)


    def test_labels(self):
        download_mnist_file(mnist_info['test']['labels'],
                            self.target_dir, force=self.force)
        return parse_labels(mnist_info['test']['labels'], self.target_dir)


    def test_images(self):
        download_mnist_file(mnist_info['test']['images'],
                            self.target_dir, force=self.force)
        return parse_images(mnist_info['test']['images'], self.target_dir)


def main():
    import matplotlib.pyplot as plt

    mnist = MNIST()

    n = 10

    Xtrain, ytrain = mnist.train_images(), mnist.train_labels()

    Xtest, ytest = mnist.test_images(), mnist.test_labels()

    rng = np.random.default_rng()

    for idx in rng.permutation(Xtrain.shape[0])[:n]:
        plt.imshow(Xtrain[idx,:,:], cmap='gray')
        plt.title('Train Set: This should be a {}!'.format(ytrain[idx]))
        plt.show()

    for idx in rng.permutation(Xtest.shape[0])[:n]:
        plt.imshow(Xtest[idx,:,:], cmap='gray')
        plt.title('Test Set: this should be a {}!'.format(ytest[idx]))
        plt.show()


if __name__ == '__main__':
    main()

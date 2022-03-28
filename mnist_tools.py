import os
import gzip
import struct
import tempfile
import shutil
import urllib.parse
import requests
import numpy as np


mnist_info = {
    'train' : { 'labels' : 'train-labels-idx1-ubyte.gz',
                'images' : 'train-images-idx3-ubyte.gz' },
    'test'  : { 'labels' : 't10k-labels-idx1-ubyte.gz',
                'images' : 't10k-images-idx3-ubyte.gz' },
    'base_url' : 'http://yann.lecun.com/exdb/mnist/'
}


idx_dt = {
    0x08 : 'B', # uint8
    0x09 : 'b', # int8
    0x0b : 'h', # int16
    0x0c : 'i', # int32
    0x0d : 'f', # float32
    0x0e : 'd', # float64
}


def download_mnist_file(fname, target_dir, force=False):
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urllib.parse.urljoin(mnist_info['base_url'], fname)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target_fname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    size = f.write(chunk)


def parse_idx(fname, target_dir):
    with gzip.open(os.path.join(target_dir, fname), 'rb') as f:
        zeros, dt, ndims = struct.unpack('>HBB', f.read(4))
        dims = struct.unpack('>' + 'I'*ndims, f.read(4*ndims))
        dt = np.dtype('>' + idx_dt[dt])
        data = np.frombuffer(f.read(), dtype=dt)
        return data.reshape(dims)


class MNIST(object):
    def __init__(self, target_dir=None, clean_up=False, force=False):
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
        return parse_idx(mnist_info['train']['labels'], self.target_dir)


    def train_images(self):
        download_mnist_file(mnist_info['train']['images'],
                            self.target_dir, force=self.force)
        return parse_idx(mnist_info['train']['images'], self.target_dir)


    def test_labels(self):
        download_mnist_file(mnist_info['test']['labels'],
                            self.target_dir, force=self.force)
        return parse_idx(mnist_info['test']['labels'], self.target_dir)


    def test_images(self):
        download_mnist_file(mnist_info['test']['images'],
                            self.target_dir, force=self.force)
        return parse_idx(mnist_info['test']['images'], self.target_dir)


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

""" Helper functions for data preprocessing """

import h5py as h5
import numpy as np
from scipy import ndimage


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def rearrange_axes(array):
    n = np.swapaxes(array, 1, 3)
    n = np.swapaxes(n, 1, 2)
    return n


def distort(X, n):
    """ Creates n samples from one image (X) one with up-down flip
    and one with left-right flip all others are randomly rotated
    from -45 to +45 degrees without reshaping
    """
    angles = (2*np.random.rand(n-2)-1) * 45
    size = X.shape
    X_distorted = np.zeros((n, size[0], size[1], size[2]))
    X_distorted[0] = np.flipud(X)
    X_distorted[1] = np.fliplr(X)
    for i in xrange(2, n):
        X_distorted[i] = ndimage.rotate(X, angles[i-2], reshape=False)
    return X_distorted


def distort_images(X, Y, n=3):
    """ Takes the images to distort in X and their corresponding labels in y

    Returns X.size[0] * n images with various distortions as shown in distort

    Note: n >= 2 (i.e. the ud and lr flip are always created)
    """
    xsize = np.array(X.shape)
    ysize = np.array(Y.shape)
    X_with_distortion = np.zeros((xsize[0]*n, xsize[1], xsize[2], xsize[3]))
    Y_with_distortion = np.zeros((ysize[0]*n, ysize[1]))
    for index, img in enumerate(X):
        X_with_distortion[n*index: n*(index+1)] = distort(img, n)
        Y_with_distortion[n*index: n*(index+1)] = Y[index]
    return X_with_distortion, Y_with_distortion


def read_data(distort=False):
    """ Reads the local file CIFAR10.hdf5
    Prepares the dataset and distorts if asked

    Returns
    -------
    - X_train: Training images
    - Y_train: Training labels
    - X_test: Testing images
    - Y_test: Testing labels
    """
    with h5.File('CIFAR10.hdf5', 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        X_test = hf.get('X_test')
        X_test = np.array(X_test)
        Y_test = hf.get('Y_test')
        Y_test = np.array(Y_test)
        Y_test = dense_to_one_hot(Y_test)
        X_train = hf.get('X_train')
        X_train = np.array(X_train)
        Y_train = hf.get('Y_train')
        Y_train = np.array(Y_train)
        Y_train = dense_to_one_hot(Y_train)
    print('Data download successful!')
    X_train, X_test = rearrange_axes(X_train), rearrange_axes(X_test)
    if distort:
        X_train, Y_train = distort_images(X_train, Y_train)
        print("Data distortion successful!")
    return X_train, Y_train, X_test, Y_test


class CIFAR10Batcher():
    def __init__(self, FLAGS):
        self.images_train, self.labels_train, \
        self.images_test, self.labels_test = read_data(distort = FLAGS.augmentation)
        self.batch_size = FLAGS.batch_size
    

    def next_batch(self, train = True):
        """Gives the data for next training/testing epoch"""
        if train:
            images, labels = self.images_train, self.labels_train
        else:
            images, labels = self.images_test, self.labels_test


        assert (len(images) == len(labels)), "Number of labels %d is different than"+\
        "number of images %d" % (len(labels), len(images))
        n = len(images)
        ids = np.random.choice(n, self.batch_size, replace = False)
        return images[ids], labels[ids]

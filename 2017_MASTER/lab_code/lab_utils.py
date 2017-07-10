import csv
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


EPS = np.finfo('float32').eps


def get_mnist_data(download_data_path, one_hot=True, verbose=False):
    """
    
    Parameters
    ----------
    download_data_path : string
        Directory where MNIST data are downloaded and extracted.
    one_hot : bool
        If True, targets are returned into one-hot format
    verbose : bool
        If True, print dataset tensors dimensions

    Returns
    -------
    mnist : Dataset
        Structure containing train, val and test mnist dataset in a friendly format.
    """

    # Download and read in MNIST dataset
    mnist = input_data.read_data_sets(download_data_path, one_hot=one_hot)

    if verbose:

        # Print image tensors shapes
        print('TRAIN tensor shape: {}'.format(mnist.train.images.shape))
        print('VAL   tensor shape: {}'.format(mnist.validation.images.shape))
        print('TEST  tensor shape: {}'.format(mnist.test.images.shape))

        # Print labels shape (encoded as one-hot vectors)
        print('TRAIN labels shape: {}'.format(mnist.train.labels.shape))
        print('VAL   labels shape: {}'.format(mnist.validation.labels.shape))
        print('TEST  labels shape: {}'.format(mnist.test.labels.shape))

    return mnist


def get_brain_body_data(csv_file):
    """
    Load brain - weight data to test linear regression.

    The data records the average weight of the brain and body for a number of mammal species.
    More details here: http://people.sc.fsu.edu/~jburkardt/datasets/regression/x01.txt

    Parameters
    ----------
    csv_file : basestring
        path of csv file containing data

    Returns
    -------
    body_weight, brain_weight : lists
        list of body and brain weight
    """
    body_weight = []
    brain_weight = []

    with open(csv_file, 'rt') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            if row:
                idx, brain_w, body_w = row[0].split()
                brain_weight.append(float(brain_w))
                body_weight.append(float(body_w))

    return body_weight, brain_weight

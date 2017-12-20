import cv2
import numpy as np
import os.path as path
import pickle
from tensorflow.examples.tutorials.mnist import input_data


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


class TilesDataset:

    def __init__(self, dataset_root):

        self.dataset_root = dataset_root

        # Store locations of train, val and test directories
        self.train_x_dir      = path.join(dataset_root, 'X_train')
        self.train_y_dir      = path.join(dataset_root, 'Y_train')
        self.validation_x_dir = path.join(dataset_root, 'X_validation')
        self.validation_y_dir = path.join(dataset_root, 'Y_validation')
        self.test_x_dir       = path.join(dataset_root, 'X_test')
        self.test_y_dir       = path.join(dataset_root, 'Y_test')

        # Number of dataset examples
        self.train_num_examples      = 10000
        self.validation_num_examples = 1000
        self.test_num_examples       = 1000

        # Initialize empty structures to contain data
        self.train_x      = []
        self.train_y      = []
        self.validation_x = []
        self.validation_y = []
        self.test_x       = []
        self.test_y       = []

        # Load images from `dataset_root`
        self._fill_data_arrays()

    def _fill_data_arrays(self):

        # Load training images
        for i in range(1, self.train_num_examples + 1):
            print('Loading training examples. {} / {}...'.format(i, self.train_num_examples))
            x_image = cv2.imread(path.join(self.train_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.train_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.train_x.append(x_image.astype(np.float32))
            self.train_y.append(np.expand_dims(y_image.astype(np.float32), 2))

        # Load validation examples
        for i in range(1, self.validation_num_examples + 1):
            print('Loading validation examples. {} / {}...'.format(i, self.validation_num_examples))
            x_image = cv2.imread(path.join(self.validation_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.validation_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.validation_x.append(x_image.astype(np.float32))
            self.validation_y.append(np.expand_dims(y_image.astype(np.float32), 2))

        # Load test examples
        for i in range(1, self.test_num_examples + 1):
            print('Loading test examples. {} / {}...'.format(i, self.test_num_examples))
            x_image = cv2.imread(path.join(self.test_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.test_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.test_x.append(x_image.astype(np.float32))
            self.test_y.append(np.expand_dims(y_image.astype(np.float32), 2))

    def dump_to_file(self, file_path, protocol=pickle.HIGHEST_PROTOCOL):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=protocol)


def convert_target_to_one_hot(target_batch):
    """
    Convert a batch of targets from (height,width,1) to (height,width,2) one-hot encoding.
    """
    b, h, w, c = target_batch.shape
    out_tensor = np.zeros(shape=(b, h, w, 2))
    for k, cur_example in enumerate(target_batch):
        foreground_mask = np.squeeze(cur_example > 0)
        background_mask = np.squeeze(cur_example == 0)
        out_tensor[k, background_mask, 0] = 1.0
        out_tensor[k, foreground_mask, 1] = 1.0
    return out_tensor

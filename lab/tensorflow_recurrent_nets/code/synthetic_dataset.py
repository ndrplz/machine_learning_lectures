import pickle
import numpy as np
from os import makedirs
from os.path import exists
from os.path import dirname
from random import shuffle


class SyntheticSequenceDataset:

    def __init__(self, dataset_cache='data/synthetic_dataset.pickle', force_recompute=False):

        self._data = None

        self.force_recompute = force_recompute
        self.dataset_cache   = dataset_cache

    @property
    def data(self):

        if not self._data:
            if not self.force_recompute and exists(self.dataset_cache):
                print('Loading dataset from cache...')
                with open(self.dataset_cache, 'rb') as dump_file:
                    dataset = pickle.load(dump_file)
            else:
                print('Recomputing dataset...')
                dataset = self._compute_dataset()
                if not exists(dirname(self.dataset_cache)):
                    makedirs(dirname(self.dataset_cache))
                with open(self.dataset_cache, 'wb') as dump_file:
                    pickle.dump(dataset, dump_file)

            # Store data
            self._data = dataset

        return self._data

    @staticmethod
    def _compute_dataset():

        n = 20
        num_examples    = 2 ** n
        num_classes     = n + 1     # there are 21 = 0, 1, ..., 20 different classes

        # How many examples to use for training (others are for test)
        num_train_examples = int(0.8 * num_examples)

        # Generate 2**20 binary strings
        data_strings = ['{0:020b}'.format(i) for i in range(num_examples)]

        # Shuffle sequences
        shuffle(data_strings)

        # Cast to numeric each generated binary string
        data_x, data_y = [], []
        for i in range(num_examples):
            train_sequence = []
            for binary_char in data_strings[i]:
                value = int(binary_char)
                train_sequence.append([value])
            data_x.append(train_sequence)           # examples are binary sequences of int {0, 1}
            data_y.append(np.sum(train_sequence))   # targets are the number of ones in the sequence

        # Convert from categorical to one-hot
        data_y_one_hot = np.eye(num_classes)[data_y]

        # Separate suggested training and test data
        train_data      = data_x[:num_train_examples]
        train_targets   = data_y_one_hot[:num_train_examples]
        test_data       = data_x[num_train_examples:]
        test_targets    = data_y_one_hot[num_train_examples:]

        return train_data, train_targets, test_data, test_targets

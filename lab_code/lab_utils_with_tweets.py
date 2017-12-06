"""
Lab utils with tweets loading. Still to make it work. 
"""

import re
import csv
import numpy as np
from collections import Counter
from tensorflow.examples.tutorials.mnist import input_data


EPS = np.finfo('float32').eps       # machine precision for float32
MAX_TWEET_CHARS = 140               # each tweet is made by max. 140 characters


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


def preprocess(line):
    """
    Pre-process a string of text. Eventually add additional pre-processing here.
    """
    line = line.lower()               # turn to lowercase
    line = line.replace('\n', '')     # remove newlines
    line = re.sub(r'\W+', ' ', line)  # keep characters only (\W is short for [^\w])

    return line


def get_dictionary(filename, dict_size=2000):
    """
    Read the tweets and return a list of the 'max_words' most common words.
    """
    all_words = []
    with open(filename, 'r') as csv_file:
        r = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in r:
            tweet = row[3]
            if len(tweet) <= MAX_TWEET_CHARS:
                words = preprocess(tweet).split()
                all_words += words

    # Make the dictionary out of only the N most common words
    word_counter = Counter(all_words)
    dictionary, _ = zip(*word_counter.most_common(min(dict_size, len(word_counter))))

    return dictionary


def load_tweet_batch(filename, batchsize, max_len=50, dict_size=2000):
    """
    Generate a batch of training data
    """

    # get the list of words that will constitute our dictionary (once only)
    dictionary = get_dictionary(filename, dict_size)

    # read training data (once only)
    rows = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in reader:
            rows.append(row)

    # prepare data structures
    X_batch = np.zeros((batchsize, max_len, len(dictionary) + 1), dtype=np.float32)
    Y_batch = np.zeros(batchsize, dtype=np.float32)

    tweet_loaded = 0
    while tweet_loaded < batchsize:

        rand_idx = np.random.randint(0, len(rows))
        Y_batch[tweet_loaded] = float(rows[rand_idx][1])

        random_tweet = rows[rand_idx][3]
        if len(random_tweet) <= MAX_TWEET_CHARS:

            words = preprocess(random_tweet).split()

            # Vectorization
            for j, w in enumerate(words):
                if j < max_len:
                    try:
                        w_idx = dictionary.index(w)
                        X_batch[tweet_loaded, j, w_idx + 1] = 1
                    except ValueError:
                        # Word not found, using the unknown
                        X_batch[tweet_loaded, j, 0] = 1

            tweet_loaded += 1

    return X_batch, Y_batch


if __name__ == '__main__':

    x_batch, y_batch = load_tweet_batch('data/tweets_train.csv', batchsize=512)

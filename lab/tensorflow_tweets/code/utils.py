"""
Lab utils with tweets loading. Still to make it work. 
"""

import re
import csv
import numpy as np
from collections import Counter
from google_drive_downloader import GoogleDriveDownloader

GoogleDriveDownloader.download_file_from_google_drive(file_id='1fHezNVY4YWJVWYb_3P3kx2e9RstjY1OK',
                                                      dest_path='data/tweets.zip',
                                                      unzip=True)

EPS = np.finfo('float32').eps       # machine precision for float32
MAX_TWEET_CHARS = 140               # each tweet is made by max. 140 characters


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


class TweetLoader(object):

    def __init__(self, filename_train, filename_val, batchsize, max_len, dict_size):

        self._filename_train = filename_train
        self._filename_val = filename_val
        self._batchsize = batchsize
        self._max_len = max_len
        self._dict_size = dict_size

        # get the list of words that will constitute our dictionary (once only)
        self._dictionary = get_dictionary(self._filename_train, dict_size)

        self._train_rows = self.read_data(self._filename_train)
        self._val_rows = self.read_data(self._filename_val)

    def read_data(self, filename):
        # read training data
        rows = []
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            for row in reader:
                rows.append(row)
        return rows

    def vectorize(self, tweet):
        words = preprocess(tweet).split()

        X = np.zeros(shape=(1, self._max_len, self._dict_size + 1))

        # Vectorization
        for j, w in enumerate(words):
            if j < self._max_len:
                try:
                    w_idx = self._dictionary.index(w)
                    X[0, j, w_idx + 1] = 1
                except ValueError:
                    # Word not found, using the unknown
                    X[0, j, 0] = 1

        return X

    def load_tweet_batch(self, mode):
        """
        Generate a batch of training data
        """
        assert mode in ['train', 'val']
        if mode == 'train':
            rows = self._train_rows
        else:
            rows = self._val_rows

        # prepare data structures
        X_batch = np.zeros((self._batchsize, self._max_len, len(self._dictionary) + 1), dtype=np.float32)
        Y_batch = np.zeros((self._batchsize, 2), dtype=np.float32)

        tweet_loaded = 0
        while tweet_loaded < self._batchsize:

            rand_idx = np.random.randint(0, len(rows))
            Y_batch[tweet_loaded, int(rows[rand_idx][1])] = 1

            random_tweet = rows[rand_idx][3]
            if len(random_tweet) <= MAX_TWEET_CHARS:

                X = self.vectorize(tweet=random_tweet)
                X_batch[tweet_loaded] = X[0]
                tweet_loaded += 1

        return X_batch, Y_batch

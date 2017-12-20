import csv
import numpy as np
from google_drive_downloader import GoogleDriveDownloader


GoogleDriveDownloader.download_file_from_google_drive(file_id='1SagLh5XNSV4znhlnkLRkV7zHPSDbOAqv',
                                                      dest_path='./data/got.zip',
                                                      unzip=True)


def gaussians_dataset(n_gaussian, n_points, mus, stds):
    """
    Provides a dataset made by several gaussians.

    Parameters
    ----------
    n_gaussian : int
        The number of desired gaussian components.
    n_points : list
        A list of cardinality of points (one for each gaussian).
    mus : list
        A list of means (one for each gaussian, e.g. [[1, 1], [3, 1]).
    stds : list
        A list of stds (one for each gaussian, e.g. [[1, 1], [2, 2]).

    Returns
    -------
    tuple
        a tuple like:
            data ndarray shape: (n_samples, dims).
            class ndarray shape: (n_samples,).
    """

    assert n_gaussian == len(mus) == len(stds) == len(n_points)

    X = []
    Y = []
    for i in range(0, n_gaussian):

        mu = mus[i]
        std = stds[i]
        n_pt = n_points[i]

        cov = np.diag(std)

        X.append(np.random.multivariate_normal(mu, cov, size=2*n_pt))
        Y.append(np.ones(shape=2*n_pt) * i)

    X = np.concatenate(X, axis=0)
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    Y = np.concatenate(Y, axis=0)

    tot = np.concatenate((X, np.reshape(Y, newshape=(-1, 1))), axis=-1)

    np.random.seed(30101990)
    np.random.shuffle(tot)
    X = tot[:, :-1]
    Y = tot[:, -1]

    # normalize X
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    n_train_samples = X.shape[0]//2

    X_train = X[:n_train_samples]
    Y_train = Y[:n_train_samples]

    X_test = X[n_train_samples:]
    Y_test = Y[n_train_samples:]

    return X_train, Y_train, X_test, Y_test


def load_got_dataset(path, train_split=0.8):
    """
    Loads the Game of Thrones dataset.

    Parameters
    ----------
    path: str
        the relative path of the csv file.
    train_split: float
        percentage of training examples in [0, 1].

    Returns
    -------
    tuple
        x_train: np.array
            training characters. shape=(n_train_examples, n_features)
        y_train: np.array
            training labels. shape=(n_train_examples,)
        train_names: np.array
            training names. shape=(n_train_examples,)
        x_test: np.array
            test characters. shape=(n_test_examples, n_features)
        y_test: np.array
            test labels. shape=(n_test_examples,)
        test_names: np.array
            test names. shape=(n_test_examples,)
        feature_names: np.array
            an array explaining each feature. shape=(n_test_examples,)
    """

    # read file into string ndarray
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array([row for row in reader])

    # extract feature names
    feature_names = data[0, 1:-1]

    # shuffle data
    data = data[1:]
    np.random.shuffle(data)

    # extract character names
    character_names = data[:, 0]

    # extract features X and targets Y
    X = np.float32(data[:, 1:-1])
    Y = np.float32(data[:, -1])

    # normalize X
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    # add bias to X
    X = np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
    feature_names = np.concatenate((feature_names, np.array(['bias'])), axis=-1)

    total_characters = X.shape[0]
    test_sampling_probs = np.ones(shape=total_characters)
    test_sampling_probs[Y == 1] /= float(np.sum(Y == 1))
    test_sampling_probs[Y == 0] /= float(np.sum(Y == 0))
    test_sampling_probs /= np.sum(test_sampling_probs)

    # sample test people without replacement
    n_test_characters = int(total_characters * (1 - train_split))
    test_idx = np.random.choice(np.arange(0, total_characters), size=(n_test_characters,),
                                replace=False, p=test_sampling_probs)
    x_test = X[test_idx]
    y_test = Y[test_idx]
    test_names = character_names[test_idx]

    # sample train people
    train_sampling_probs = test_sampling_probs.copy()
    train_sampling_probs[test_idx] = 0
    train_sampling_probs /= np.sum(train_sampling_probs)

    n_train_characters = int(total_characters * train_split)
    train_idx = np.random.choice(np.arange(0, total_characters), size=(n_train_characters,),
                                 replace=True, p=train_sampling_probs)
    x_train = X[train_idx]
    y_train = Y[train_idx]
    train_names = character_names[train_idx]

    return x_train, y_train, train_names, x_test, y_test, test_names, feature_names


if __name__ == '__main__':
    A = load_got_dataset(path='x.csv')

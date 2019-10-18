import numpy as np
from sklearn.datasets import make_moons
from google_drive_downloader import GoogleDriveDownloader


GoogleDriveDownloader.download_file_from_google_drive(file_id='1IrEXRIVUPANdS6syggZyR7bABa7pCKjQ',
                                                      dest_path='./data/boosting.zip',
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
            X_train ndarray shape: (n_samples, dims).
            Y_train ndarray shape: (n_samples,).
            X_test ndarray shape: (n_samples, dims).
            Y_test ndarray shape: (n_samples,).
    """

    assert n_gaussian == len(mus) == len(stds) == len(n_points)

    X = []
    Y = []
    for i in range(n_gaussian):

        mu = mus[i]
        std = stds[i]
        n_pt = n_points[i]

        cov = np.diag(std)

        X.append(np.random.multivariate_normal(mu, cov, size=2*n_pt))
        Y.append(np.ones(shape=2*n_pt) * i)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    tot = np.concatenate((X, np.reshape(Y, newshape=(-1, 1))), axis=-1)

    np.random.seed(30101990)
    np.random.shuffle(tot)
    X = tot[:, :-1]
    Y = tot[:, -1]

    n_train_samples = X.shape[0]//2

    X_train = X[:n_train_samples]
    Y_train = Y[:n_train_samples]

    X_test = X[n_train_samples:]
    Y_test = Y[n_train_samples:]

    Y_train[Y_train == 0] = -1
    Y_test[Y_test == 0] = -1

    return X_train, Y_train, X_test, Y_test


def two_moon_dataset(n_samples=100, shuffle=True, noise=None, random_state=None):
    """
    Make two interleaving half circles

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    Read more in the :ref:`User Guide <sample_generators>`.

    Returns
    -------
    tuple
        a tuple like:
            X_train ndarray shape: (n_samples, dims).
            Y_train ndarray shape: (n_samples,).
            X_test ndarray shape: (n_samples, dims).
            Y_test ndarray shape: (n_samples,).
    """
    X_train, Y_train = make_moons(n_samples, shuffle, noise, random_state)
    X_test, Y_test = make_moons(n_samples, shuffle, noise, random_state)

    Y_train[Y_train == 0] = -1
    Y_test[Y_test == 0] = -1

    return X_train, Y_train, X_test, Y_test


def h_shaped_dataset():
    """
    Yet another dataset to experiment with boosting.
    It returns a complex non-linear binary dataset.

    Returns
    -------
    tuple
        a tuple like:
            X_train ndarray shape: (n_samples, dims).
            Y_train ndarray shape: (n_samples,).
            X_test ndarray shape: (n_samples, dims).
            Y_test ndarray shape: (n_samples,).
    """

    data = np.load('data/data.npy')
    labels = np.squeeze(np.load('data/labels.npy'))

    # shuffle
    n, d  = data.shape
    idx = np.arange(0, n)
    np.random.shuffle(idx)

    X_train = data[idx[:n//2]]
    Y_train = labels[idx[:n // 2]]

    X_test = data[idx[n//2:]]
    Y_test = labels[idx[n//2:]]

    return X_train, Y_train, X_test, Y_test

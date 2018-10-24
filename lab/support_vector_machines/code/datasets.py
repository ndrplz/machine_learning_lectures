import cv2
import numpy as np
from glob import glob
from os.path import join
from sklearn.datasets import make_moons
from google_drive_downloader import GoogleDriveDownloader

GoogleDriveDownloader.download_file_from_google_drive(file_id='1hM_kk3ys2YnaZbIBwwdXAMhJm4j9KaKI',
                                                      dest_path='./data/svm.zip',
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
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """
    X_train, Y_train = make_moons(n_samples, shuffle, noise, random_state)
    X_test, Y_test = make_moons(n_samples, shuffle, noise, random_state)

    return X_train, Y_train, X_test, Y_test


def people_dataset(data_path, train_split=60):
    """
    Function that loads data for people vs non people classification.
    
    Parameters
    ----------
    data_path: str
        the dataset root folder.
    train_split: int
        percentage of points for training set (default is 60%).

    Returns
    -------
    tuple
        A tuple like (X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test)

    """
    X_img = []
    Y = []
    X_feat = []

    for l, c in enumerate(['non_people', 'people']):
        img_list = glob(join(data_path, c, '*.pgm'))
        X_img.append(np.array([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in img_list]))
        Y.append(np.ones(shape=len(img_list)) * l)
        X_feat.append(np.load(join(data_path, c + '.npy')))

    X_img = np.concatenate(X_img, axis=0)
    X_feat = np.concatenate(X_feat, axis=0)
    Y = np.concatenate(Y, axis=0)

    idx = np.arange(0, X_img.shape[0])
    np.random.shuffle(idx)

    X_img = X_img[idx]
    X_feat = X_feat[idx]
    Y = Y[idx]

    n_train_samples = X_img.shape[0] * train_split // 100

    X_img_train = X_img[:n_train_samples]
    X_feat_train = X_feat[:n_train_samples]
    Y_train = Y[:n_train_samples]

    X_img_test = X_img[n_train_samples:]
    X_feat_test = X_feat[n_train_samples:]
    Y_test = Y[n_train_samples:]

    return X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test

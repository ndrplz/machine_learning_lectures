"""
Functions to load data from file.
"""

import numpy as np
import skimage.io as io
from os.path import join, basename, isdir
from glob import glob
from google_drive_downloader import GoogleDriveDownloader


GoogleDriveDownloader.download_file_from_google_drive(file_id='1SpyIe1jwiFV4s5ulinHiW2CfviPN06er',
                                                      dest_path='./att_faces/eigenfaces.zip',
                                                      unzip=True)


def get_faces_dataset(path, train_split=60):
    """
    Loads Olivetti dataset from files.
    
    Parameters
    ----------
    path: str
        the root folder of the Olivetti dataset.
    train_split: int
        the percentage of dataset used for training (default is 60%).

    Returns
    -------
    tuple
        a tuple like (X_train, Y_train, X_test, Y_test)
    """

    cl_folders = sorted([basename(f) for f in glob(join(path, '*')) if isdir(f)])

    X = []
    Y = []
    for cl, cl_f in enumerate(cl_folders):
        img_list = glob(join(path, cl_f, '*.pgm'))

        for i, img_path in enumerate(img_list):
            X.append(io.imread(img_path, as_grey=True).ravel())
            Y.append(cl)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    n_samples = Y.size
    n_train_samples = (n_samples * train_split) // 100

    # shuffle
    tot = np.concatenate((X, np.reshape(Y, newshape=(-1, 1))), axis=-1)

    np.random.seed(30101990)
    np.random.shuffle(tot)
    X = tot[:, :-1]
    Y = tot[:, -1]

    X_train = X[:n_train_samples]
    Y_train = Y[:n_train_samples]

    X_test = X[n_train_samples:]
    Y_test = Y[n_train_samples:]

    return X_train, Y_train, X_test, Y_test

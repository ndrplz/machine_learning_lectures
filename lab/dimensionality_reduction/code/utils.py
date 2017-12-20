"""
Some plotting functions.
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


plt.ion()


def show_eigenfaces(eigenfaces, size):
    """
    Plots ghostly eigenfaces.
    
    Parameters
    ----------
    eigenfaces: ndarray
        eigenfaces (eigenvectors of face covariance matrix).
    size: tuple
        the size of each face image like (h, w).

    Returns
    -------
    None
    """
    eigf = []
    for f in eigenfaces.T.copy():

        f -= f.min()
        f /= f.max() + np.finfo(float).eps

        eigf.append(np.reshape(f, newshape=size))

    to_show = np.concatenate(eigf, axis=1)

    plt.imshow(to_show)
    plt.title('Eigenfaces')
    plt.waitforbuttonpress()


def show_3d_faces_with_class(points, labels):
    """
    Plots 3d data in colorful point (color is class).
    
    Parameters
    ----------
    points: ndarray
        3d points to plot (shape: (n_samples, 3)).
    labels: ndarray
        classes (shape: (n_samples,)).

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5, c=labels, s=60)
    plt.show(block=True)

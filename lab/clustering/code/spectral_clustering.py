import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from numpy.linalg import eigh

from scipy.linalg import fractional_matrix_power

from kmeans_clustering import kmeans


import matplotlib.pyplot as plt
plt.ion()


def spectral_clustering(data, n_cl, sigma=1.):
    """
    Spectral clustering.
    
    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    sigma: float
        std of radial basis function kernel.

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    n_samples, dim = data.shape

    # build adjacency matrix
    adj = np.zeros(shape=(n_samples, n_samples))
    for i in xrange(0, n_samples):
        x1 = data[i]
        for j in xrange(i, n_samples):
            x2 = data[j]
            adj[i, j] = np.exp(- np.sum(np.square(x1 - x2)) / (sigma ** 2))

    adj += adj.T
    adj -= np.eye(n_samples)

    # build degree matrix
    deg = np.diag(np.sum(adj, axis=1))

    # laplacian matrix
    deg_ = fractional_matrix_power(deg, -0.5)
    lap = np.eye(n_samples)-deg_.dot(adj).dot(deg_)

    # compute eigenvalues and eigenvectors
    eigval, eigvec = eigh(lap)
    idx = eigval.argsort()
    eigvec = eigvec[:, idx]

    to_cluster = eigvec[:, 1:n_cl]

    labels = kmeans(to_cluster, n_cl, verbose=False)

    return labels


def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """
    # generate the dataset
    data, cl = two_moon_dataset(n_samples=300, noise=0.1)

    # visualize the dataset
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)
    plt.waitforbuttonpress()

    # run spectral clustering
    labels = spectral_clustering(data, n_cl=2, sigma=0.1)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_spectral_clustering()

"""
Eigenfaces main script.
"""

import numpy as np

from utils import show_eigenfaces, show_3d_faces_with_class
from data_io import get_faces_dataset

import matplotlib.pyplot as plt
plt.ion()


def eigenfaces(X, n_comp):
    """
    Performs PCA to project faces in a reduced space.

    Parameters
    ----------
    X: ndarray
        faces to project (shape: (n_samples, w*h))
    n_comp: int
        number of principal components

    Returns
    -------
    tuple
        proj_faces: the projected faces shape=(n_samples, n_comp).
        ef: eigenfaces (the principal directions)
    """

    n_samples, dim = X.shape

    # compute mean vector
    X_mean = np.mean(X, axis=0)

    # show mean face
    plt.imshow(np.reshape(X_mean, newshape=(112, 92)))
    plt.title('mean face')
    plt.waitforbuttonpress()

    # normalize data (remove mean)
    X_norm = X - X_mean

    # trick (transpose data matrix)
    X_norm = X_norm.T

    # compute covariance
    cov = np.dot(X_norm.T, X_norm)

    # compute (sorted) eigenvectors of the covariance matrix
    eigval, eigvec = np.linalg.eig(cov)
    idxs = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, idxs]
    eigvec = eigvec[:, 0:n_comp]

    # retrieve original eigenvec
    ef = np.dot(X.T, eigvec)

    # show eigenfaces
    show_eigenfaces(ef, (112, 92))

    # project faces according to the computed directions
    proj_faces = np.dot(X, ef)

    return proj_faces, ef


def main():
    """
    Main function.
    """

    # number of principal components
    n_comp = 10

    # get_data
    X_train, Y_train, X_test, Y_test = get_faces_dataset(path='att_faces')

    proj_train, ef = eigenfaces(X_train, n_comp=n_comp)

    # visualize projections if 3d
    if n_comp == 3:
        show_3d_faces_with_class(proj_train, Y_train)

    # project test data
    test_proj = np.dot(X_test, ef)

    # predict test faces
    predictions = np.zeros_like(Y_test)
    nearest_neighbors = np.zeros_like(Y_test, dtype=np.int32)
    for i in range(0, test_proj.shape[0]):

        cur_test = test_proj[i]

        distances = np.sum(np.square(proj_train - cur_test), axis=1)

        # nearest neighbor classification
        nearest_neighbor = np.argmin(distances)
        nearest_neighbors[i] = nearest_neighbor
        predictions[i] = Y_train[nearest_neighbor]

    print('Error: {}'.format(float(np.sum(predictions != Y_test))/len(predictions)))

    # visualize nearest neighbors
    _, (ax0, ax1) = plt.subplots(1, 2)
    while True:

        # extract random index
        test_idx = np.random.randint(0, X_test.shape[0])

        ax0.imshow(np.reshape(X_test[test_idx], newshape=(112, 92)), cmap='gray')
        ax0.set_title('Test face')
        ax1.imshow(np.reshape(X_train[nearest_neighbors[test_idx]], newshape=(112, 92)), cmap='gray')
        ax1.set_title('Nearest neighbor')

        # plot faces
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()

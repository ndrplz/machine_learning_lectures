"""
Eigenfaces main script.
"""

import numpy as np

from utils import show_eigenfaces
from utils import show_nearest_neighbor
from data_io import get_faces_dataset

import matplotlib.pyplot as plt
plt.ion()

class Eigenfaces:
    """
    Performs PCA to project faces in a reduced space.
    """

    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components: int
            number of principal component
        """
        self.n_components = n_components

        # Per-feature empirical mean, estimated from the training set.
        self._x_mean = None

        #Principal axes in feature space, representing the directions
        # of maximum variance in the data.
        self._ef = None

    def fit(self, X: np.ndarray, verbose: bool = False):
        """
        Parameters
        ----------
        X: ndarray
            Training set will be used for fitting the PCA
                (shape: (n_samples, w*h))
        verbose: bool
        """

        # compute mean vector and store it for the inference stage
        self._x_mean = np.mean(X, axis=0)

        if verbose:
            # show mean face
            plt.imshow(np.reshape(self._x_mean, newshape=(112, 92)), cmap='gray')
            plt.title('mean face')
            plt.waitforbuttonpress()
            plt.close()

        # normalize data (remove mean)
        X_norm = X - self._x_mean

        # Eigen-trick (transpose data matrix)
        X_norm = X_norm.T

        # compute covariance
        cov = np.dot(X_norm.T, X_norm)

        # compute (sorted) eigenvectors of the covariance matrix
        eigval, eigvec = np.linalg.eig(cov)
        idxs = np.argsort(eigval)[::-1]
        eigvec = eigvec[:, idxs]
        eigvec = eigvec[:, 0:self.n_components]

        # retrieve original eigenvec
        self._ef = np.dot(X.T, eigvec)

        # normalize the retrieved eigenvectors to have unit length
        self._ef = self._ef / np.sqrt(eigval[idxs][0:self.n_components])

        if verbose:
            # show eigenfaces
            show_eigenfaces(self._ef, (112, 92))
            plt.waitforbuttonpress()
            plt.close()

    def transform(self, X: np.ndarray):
        """
        Parameters
        ----------
        X: ndarray
            faces to project (shape: (n_samples, w*h))
        Returns
        -------
        out: ndarray
            projections (shape: (n_samples, self.n_components)
        """

        # project faces according to the computed directions
        return np.dot(X-self._x_mean, self._ef)

    def inverse_transform(self, X: np.ndarray):
        return np.dot(X, self._ef.T) + self._x_mean


class NearestNeighbor:

    def __init__(self):
        self._X_db, self._Y_db = None, None

    def fit(self, X: np.ndarray, y: np.ndarray,):
        """
        Fit the model using X as training data and y as target values
        """
        self._X_db = X
        self._Y_db = y

    def predict(self, X: np.ndarray):
        """
        Finds the 1-neighbor of a point. Returns predictions as well as indices of
        the neighbors of each point.
        """
        num_test_samples = X.shape[0]

        # predict test faces
        predictions = np.zeros((num_test_samples,))
        nearest_neighbors = np.zeros((num_test_samples,), dtype=np.int32)

        for i in range(num_test_samples):

            distances = np.sum(np.square(self._X_db - X[i]), axis=1)

            # nearest neighbor classification
            nearest_neighbor = np.argmin(distances)
            nearest_neighbors[i] = nearest_neighbor
            predictions[i] = self._Y_db[nearest_neighbor]

        return predictions, nearest_neighbors

def main():

    # get_data
    X_train, Y_train, X_test, Y_test = get_faces_dataset(path='att_faces', train_split=0.6)

    # number of principal components
    n_components = 30

    # fit the PCA transform
    eigpca = Eigenfaces(n_components)
    eigpca.fit(X_train, verbose=True)

    # project the training data
    proj_train = eigpca.transform(X_train)

    # project the test data
    proj_test = eigpca.transform(X_test)

    # fit a 1-NN classifier on PCA features
    nn = NearestNeighbor()
    nn.fit(proj_train, Y_train)

    # Compute predictions and indices of 1-NN samples for the test set
    predictions, nearest_neighbors = nn.predict(proj_test)

    # Compute the accuracy on the test set
    test_set_accuracy = float(np.sum(predictions == Y_test))/len(predictions)
    print(f'Test set accuracy: {test_set_accuracy}')

    # Show results.
    show_nearest_neighbor(X_train, Y_train,
                          X_test, Y_test, nearest_neighbors)


if __name__ == '__main__':
    main()


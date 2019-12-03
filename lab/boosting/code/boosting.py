import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice


class WeakClassifier:
    """
    Class that models a WeakClassifier
    """
    def __init__(self):
        self._dim = None
        self._threshold = None
        self._label_above_split = None

    def fit(self, X: np.ndarray, Y: np.ndarray):

        # Select random feature (see np.random.choice)
        _, n_feats = X.shape
        self._dim = choice(a=range(0, n_feats))

        # Select random split threshold
        feat_min = np.min(X[:, self._dim])
        feat_max = np.max(X[:, self._dim])
        self._threshold = np.random.uniform(low=feat_min, high=feat_max)

        # Select random verse
        possible_labels = np.unique(Y)
        self._label_above_split = choice(a=possible_labels)

    def predict(self, X: np.ndarray):
        y_pred = np.zeros(shape=X.shape[0])
        y_pred[X[:, self._dim] >= self._threshold] = self._label_above_split
        y_pred[X[:, self._dim] < self._threshold] = -1 * self._label_above_split

        return y_pred


class AdaBoostClassifier:
    """
    Class encapsulating AdaBoost classifier
    """
    def __init__(self, n_learners: int, n_max_trials: int = 200):
        """
        Initialize an AdaBoost classifier.

        Parameters
        ----------
        n_learners: int
            Number of weak classifiers.
        """
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)

        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Trains the model.

        Parameters
        ----------
        X: np.ndarray
            Features having shape (n_samples, dim).
        Y: np.ndarray
            Class labels having shape (n_samples,).
        verbose: bool
            Whether or not to visualize the learning process (default=False).
        """

        n_examples, n_feats = X.shape

        distinct_labels = len(np.unique(Y))
        if distinct_labels == 1:
            warnings.warn('Fitting {} on a dataset with only one label.'.format(
                self.__class__.__name__))
        elif distinct_labels > 2:
            raise NotImplementedError('Only binary classification is supported.')

        # Initialize all examples with equal weights
        weights = np.ones(shape=n_examples) / n_examples

        # Train ensemble
        for l in range(self.n_learners):
            # Perform a weighted re-sampling (with replacement) of the dataset
            #  to create a new dataset on which the current weak learner will
            #  be trained.
            sampled_idxs = choice(a=range(0, n_examples), size=n_examples,
                                  replace=True, p=weights)
            cur_X = X[sampled_idxs]
            cur_Y = Y[sampled_idxs]

            # Search for a weak classifier
            n_trials = 0
            error = 1.
            while error > 0.5:
                weak_learner = WeakClassifier()
                weak_learner.fit(cur_X, cur_Y)
                y_pred = weak_learner.predict(cur_X)

                # Compute current weak learner error
                error = np.sum(weights[sampled_idxs[cur_Y != y_pred]])

                # Re-initialize sample weights if number of trials is exceeded
                n_trials += 1
                if n_trials > self.n_max_trials:
                    weights = np.ones(shape=n_examples) / n_examples

            # Store weak learner parameter
            self.alphas[l] = alpha = np.log((1 - error) / error) / 2

            # Append the weak classifier to the chain
            self.learners.append(weak_learner)

            # Update examples weights
            weights[sampled_idxs[cur_Y != y_pred]] *= np.exp(alpha)
            weights[sampled_idxs[cur_Y == y_pred]] *= np.exp(-alpha)
            weights /= np.sum(weights)  # re-normalize

            # Possibly plot the predictions (if these are 2D)
            if verbose and n_feats == 2:
                self._plot(cur_X, y_pred, weights[sampled_idxs],
                           self.learners[-1], l)

    def predict(self, X: np.ndarray):
        """
        Function to perform predictions over a set of samples.

        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """
        num_samples = X.shape[0]

        pred_all_learners = np.zeros(shape=(num_samples, self.n_learners))

        for l, learner in enumerate(self.learners):
            pred_all_learners[:, l] = learner.predict(X)

        # weight for learners efficiency
        pred_all_learners *= self.alphas

        # compute predictions
        pred = np.sign(np.sum(pred_all_learners, axis=1))

        return pred

    @staticmethod
    def _plot(X: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
              learner: WeakClassifier, iteration: int, cmap: str = 'jet'):

        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=weights * 50000,
                    cmap=cmap, edgecolors='k')

        M1, m1 = np.max(X[:, 1]), np.min(X[:, 1])
        M0, m0 = np.max(X[:, 0]), np.min(X[:, 0])

        cur_split = learner._threshold
        if learner._dim == 0:
            plt.plot([cur_split, cur_split], [m1, M1], 'k-', lw=5)
        else:
            plt.plot([m0, M0], [cur_split, cur_split], 'k-', lw=5)
        plt.xlim([m0, M0])
        plt.ylim([m1, M1])
        plt.xticks([])
        plt.yticks([])
        plt.title('Iteration: {:04d}'.format(iteration))
        plt.waitforbuttonpress(timeout=0.1)

import matplotlib.pyplot as plt
import numpy as np

from utils import cmap


class WeakClassifier:
    """
    Function that models a WeakClassifier
    """

    def __init__(self):

        # initialize a few stuff
        self._dim = None
        self._threshold = None
        self._label_above_split = None

    def fit(self, X: np.ndarray, Y: np.ndarray):

        n, d = X.shape
        possible_labels = np.unique(Y)

        # select random feature (see np.random.choice)
        self._dim = np.random.choice(a=range(0, d))

        # select random split (see np.random.uniform)
        M, m = np.max(X[:, self._dim]), np.min(X[:, self._dim])
        self._threshold = np.random.uniform(low=m, high=M)

        # select random verse (see np.random.choice)
        self._label_above_split = np.random.choice(a=possible_labels)

    def predict(self, X: np.ndarray):

        num_samples = X.shape[0]
        y_pred = np.zeros(shape=num_samples)
        y_pred[X[:, self._dim] >= self._threshold] = self._label_above_split
        y_pred[X[:, self._dim] < self._threshold] = -1 * self._label_above_split

        return y_pred


class AdaBoostClassifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners: int, n_max_trials: int = 200):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of weak classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)

        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        # some inits
        n, d = X.shape
        if d != 2:
            verbose = False  # only plot learning if 2 dimensional

        possible_labels = np.unique(Y)

        # only binary problems please
        assert possible_labels.size == 2, 'Error: data is not binary'

        # initialize the sample weights as equally probable
        sample_weights = np.ones(shape=n) / n

        # start training
        for l in range(self.n_learners):

            # choose the indexes of 'difficult' samples (np.random.choice)
            cur_idx = np.random.choice(a=range(0, n), size=n, replace=True, p=sample_weights)

            # extract 'difficult' samples
            cur_X = X[cur_idx]
            cur_Y = Y[cur_idx]

              # search for a weak classifier
            error = 1
            n_trials = 0
            cur_wclass = None
            y_pred = None

            while error > 0.5:

                cur_wclass = WeakClassifier()
                cur_wclass.fit(cur_X, cur_Y)
                y_pred = cur_wclass.predict(cur_X)

                # compute error
                error = np.sum(sample_weights[cur_idx[cur_Y != y_pred]])

                n_trials += 1
                if n_trials > self.n_max_trials:
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n

            # save weak learner parameter
            self.alphas[l] = alpha = np.log((1 - error) / error) / 2

            # append the weak classifier to the chain
            self.learners.append(cur_wclass)

            # update sample weights
            sample_weights[cur_idx[cur_Y != y_pred]] *= np.exp(alpha)
            sample_weights[cur_idx[cur_Y == y_pred]] *= np.exp(-alpha)
            sample_weights /= np.sum(sample_weights)

            if verbose:
                self._plot(cur_X, y_pred, sample_weights[cur_idx],
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

    def _plot(self, X: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
              learner: WeakClassifier, iteration: int):

        # plot
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

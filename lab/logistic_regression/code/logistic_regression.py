import numpy as np


eps = np.finfo(float).eps


def sigmoid(x):
    """
    Element-wise sigmoid function

    Parameters
    ----------
    x: np.array
        a numpy array of any shape

    Returns
    -------
    np.array
        an array having the same shape of x.
    """

    return 1 / (1 + np.exp(-x))


def loss(y_true, y_pred):
    """
    The binary crossentropy loss.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)

    Returns
    -------
    float
        the value of the binary crossentropy.
    """

    return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def dloss_dw(y_true, y_pred, X):
    """
    Derivative of loss function w.r.t. weights.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)
    X: np.array
        predicted data. shape=(n_examples, n_features)

    Returns
    -------
    np.array
        derivative of loss function w.r.t weights.
        Has shape=(n_features,)
    """

    N = y_true.shape[0]
    return - np.dot(X.T, (y_true - y_pred)) / N


class LogisticRegression:
    """ Models a logistic regression classifier. """

    def __init__(self):
        """ Constructor method """

        # weights placeholder
        self._w = None

    def fit_gd(self, X, Y, n_epochs, learning_rate, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        n_epochs: int
            number of gradient updates.
        learning_rate: float
            step towards the descent.
        verbose: bool
            whether or not to print the value of cost function.
        """

        n_samples, n_features = X.shape

        # initialize weights
        self._w = np.random.normal(loc=0, scale=0.001, size=(n_features,))

        # loop over epochs
        for e in range(0, n_epochs):

            # predict training data
            cur_prediction = sigmoid(np.dot(X, self._w))

            # compute (and print) cost
            cur_loss = loss(y_true=Y, y_pred=cur_prediction)
            if verbose:
                print(cur_loss)

            # update weights following gradient
            self._w -= learning_rate * dloss_dw(y_true=Y, y_pred=cur_prediction, X=X)

    def predict(self, X):
        """
        Function that predicts.

        Parameters
        ----------
        X: np.array
            data to be predicted. shape=(n_test_examples, n_features)

        Returns
        -------
        prediction: np.array
            prediction in {0, 1}.
            Shape is (n_test_examples,)
        """

        # predict testing data
        prediction = np.round(sigmoid(np.dot(X, self._w)))

        return prediction

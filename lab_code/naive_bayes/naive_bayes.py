"""
Class that models a Naive Bayes Classifier
"""

import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier.
    Training:
    For each class, a naive likelyhood model is estimated for P(X/Y),
    and the prior probability P(Y) is computed.
    Inference:
    performed according with the Bayes rule:
    P = argmax_Y (P(X/Y) * P(Y))
    or
    P = argmax_Y (log(P(X/Y)) + log(P(Y)))
    """

    def __init__(self):
        """
        Class constructor
        """

        self._classes = None
        self._n_classes = 0

        self._eps = np.finfo(np.float32).eps

        # array of classes prior probabilities
        self._class_priors = []

        # array of probabilities of a pixel being active (for each class)
        self._pixel_probs_given_class = []

    def fit(self, X, Y):
        """
        Computes, for each class, a naive likelyhood model (self._pixel_probs_given_class),
        and a prior probability (self.class_priors).
        Both quantities are estimated from examples X and Y.

        Parameters
        ----------
        X: np.array
            input MNIST digits. Has shape (n_train_samples, h, w)
        Y: np.array
            labels for MNIST digits. Has shape (n_train_samples,)
        """

        n_train_samples, h, w = X.shape

        self._classes = np.unique(Y)
        self._n_classes = len(self._classes)

        # compute prior and pixel probabilities for each class
        for c_idx, c in enumerate(self._classes):

            # examples of this class
            x_c = X[Y == c]

            # prior probability
            prior_c = np.sum(np.uint8(Y == c)) / n_train_samples
            self._class_priors.append(prior_c)

            probs_c = self._estimate_pixel_probabilities(x_c)
            self._pixel_probs_given_class.append(probs_c)

    def predict(self, X):
        """
        Performs inference on test data.
        Inference is performed according with the Bayes rule:
        P = argmax_Y (log(P(X/Y)) + log(P(Y)) - log(P(X)))

        Parameters
        ----------
        X: np.array
            MNIST test images. Has shape (n_test_samples, h, w).

        Returns
        -------
        prediction: np.array
            model predictions over X. Has shape (n_test_samples,)
        """

        n_test_samples, h, w = X.shape

        # initialize log probabilities of class
        class_log_probs = np.zeros(shape=(n_test_samples, self._n_classes))

        for c in range(0, self._n_classes):

            # extract class models
            pix_probs_c = self._pixel_probs_given_class[c]
            prior_c = self._class_priors[c]

            # prior probability of this class
            log_prior_c = np.log(prior_c)

            # likelyhood of examples given class
            log_lk_x = self.get_log_likelyhood_under_model(X, pix_probs_c)

            # bayes rule for logarithm
            log_prob_c = log_lk_x + log_prior_c

            # set class probability for each test example
            class_log_probs[:, c] = log_prob_c

        # class_log_probs -= np.log(np.sum(np.exp(class_log_probs), axis=1, keepdims=True))

        predictions = np.argmax(class_log_probs, axis=1)

        return predictions

    @staticmethod
    def _estimate_pixel_probabilities(images):
        """
        Estimates pixel probabilities from data.

        Parameters
        ----------
        images: np.array
            images to estimate pixel probabilities from. Has shape (n_images, h, w)

        Returns
        -------
        pix_probs: np.array
            probabilities for each pixel of being 1, estimated from images.
            Has shape (h, w)
        """

        pix_probs = np.mean(images, axis=0)
        return pix_probs

    def get_log_likelyhood_under_model(self, images, model):
        """
        Returns the likelyhood of many images under a certain model.
        Naive:
        the likelyhood of the image is the product of the likelyhood of each pixel.
        or
        the log-likelyhood of the image is the sum of the log-likelyhood of each pixel.

        Parameters
        ----------
        images: np.array
            input images. Having shape (n_images, h, w).
        model: np.array
            a model of pixel probabilities, having shape (h, w)

        Returns
        -------
        lkl: np.array
            the likelyhood of each pixel under the model, having shape (h, w).
        """
        n_samples = images.shape[0]

        model = np.tile(np.expand_dims(model, axis=0), reps=(n_samples, 1, 1))

        idx_1 = (images == 1)
        idx_0 = (images == 0)

        lkl = np.zeros_like(images, dtype=np.float32)
        lkl[idx_1] = model[idx_1]
        lkl[idx_0] = 1 - model[idx_0]

        log_lkl = np.apply_over_axes(np.sum, np.log(lkl + self._eps), axes=[1, 2])
        log_lkl = np.squeeze(log_lkl)

        return log_lkl

import numpy as np

class SVM:

    CLASS_0_INT = -1.0
    CLASS_1_INT = 1.0

    def __init__(self,n_epochs,lam):
        """
            n_epochs: int
                number of gradient updates.
            lam: float
                regularization term coeff.
        """
        self._w, self.unique_y = None, None
        self._n_epochs = n_epochs
        self._lambda = lam

    def set_groundtruth_mapping(self, y):
        self._unique_y = np.unique(y)
        assert self._unique_y.shape[0] == 2

    def map_y_to_negative_plus_one(self, y):
        vfunc = np.vectorize(lambda t: SVM.CLASS_0_INT if t == self._unique_y[0] else SVM.CLASS_1_INT)
        return vfunc(y)

    def map_y_to_original_values(self, y):
        vfunc = np.vectorize(lambda t: self._unique_y[0] if t == SVM.CLASS_0_INT else self._unique_y[1])
        return vfunc(y)

    def loss(self, y_true, y_pred):
        """
            The PEGASOS loss, define as the sum of the regularization term
            and the hinge loss w.r.t. a desired margin equal to one.

            Parameters
            ----------
            y_true: np.array
                real labels in {0, 1}. shape=(n_examples,)
            y_pred: np.array
                predicted labels in [0, 1]. shape=(n_examples,)
            Returns
            -------
            float
                the value of the PEGASOS loss.
        """
        return 0.5*self._lambda*np.linalg.norm(self._w) + \
               np.mean(np.maximum(np.zeros(y_true.shape), 1.0 - y_true*y_pred))

    def fit_gd(self, X, Y, verbose=False):
        """
            Implements the gradient descent training procedure.

            Parameters
            ----------
            X: np.array
                data. shape=(n_examples, n_features)
            Y: np.array
                labels. shape=(n_examples,)
            verbose: bool
                whether or not to print the value of cost function.
        """

        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)

        n_samples, n_features = X.shape
        self.set_groundtruth_mapping(Y)
        self._w = np.zeros(shape=(n_features,))
        Ynpone = self.map_y_to_negative_plus_one(Y)

        t = 0
        for epoch in range(self._n_epochs+1):
            if verbose:
                print("Epoch: {} loss : {}".format(epoch, self.loss(Ynpone,y_pred=np.dot(X, self._w))))

            for j in range(n_samples):
                t += 1
                n_t = 1.0 / (t * self._lambda)
                X_j, y_j = X[j,:], Ynpone[j]
                cur_prediction = np.dot(X_j, self._w)
                if y_j * cur_prediction < 1.0:
                    self._w = (1.0 - n_t * self._lambda) * self._w + n_t * y_j * X_j
                else:
                    self._w = (1.0 - n_t * self._lambda) * self._w

    def predict(self, X):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)
        return self.map_y_to_original_values(np.where(np.dot(X, self._w) > 0.0, SVM.CLASS_1_INT, SVM.CLASS_0_INT))
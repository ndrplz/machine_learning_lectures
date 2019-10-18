import matplotlib.pyplot as plt
import numpy as np
from datasets import gaussians_dataset
from utils import plot_2d_dataset
from utils import plot_boundary

from boosting import AdaBoostClassifier

plt.ion()

def main_adaboost():
    """
    Main function for testing Adaboost.
    """

    X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [300, 400], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    # X_train, Y_train, X_test, Y_test = h_shaped_dataset()
    # X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=500, noise=0.2)

    # visualize dataset
    plot_2d_dataset(X_train, Y_train, 'Training')

    # train model and predict
    model = AdaBoostClassifier(n_learners=100)

    model.fit(X_train, Y_train, verbose=True)
    P = model.predict(X_test)

    # visualize the boundary!
    plot_boundary(X_train, Y_train, model)

    # evaluate and print error
    error = float(np.sum(P == Y_test)) / Y_test.size
    print('Test set - Classification Accuracy: {}'.format(error))


# entry point
if __name__ == '__main__':
    main_adaboost()

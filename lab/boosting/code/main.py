import matplotlib.pyplot as plt
import numpy as np

from boosting import AdaBoostClassifier
from datasets import gaussians_dataset
from utils import plot_2d_dataset
from utils import plot_boundary


def main_adaboost():
    """
    Main function for fitting and testing Adaboost classifier.
    """
    X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [300, 400], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    # X_train, Y_train, X_test, Y_test = h_shaped_dataset()
    # X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=500, noise=0.2)

    # Visualize dataset
    plot_2d_dataset(X_train, Y_train, 'Training', blocking=False)

    # Init model
    model = AdaBoostClassifier(n_learners=100)

    # Train
    model.fit(X_train, Y_train, verbose=True)

    # Predict
    y_preds = model.predict(X_test)
    print('Accuracy on test set: {}'.format(np.mean(y_preds == Y_test)))

    # Visualize the predicted boundary
    plot_boundary(X_train, Y_train, model)


if __name__ == '__main__':

    plt.ion()

    main_adaboost()

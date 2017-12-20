import numpy as np
from sklearn import svm

from datasets import gaussians_dataset, two_moon_dataset, people_dataset
from utils import plot_margin, visualize_predictions


def main_svm():
    """
    Main function to experiment with SVM on synthetic points.
    """

    X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    # X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=300, noise=0.2)

    C = 100
    kernel = 'rbf'

    model = svm.SVC(C=C, kernel=kernel)
    model.fit(X_train, Y_train)

    # print result on train
    plot_margin(X_train, Y_train, model, title='train data')

    # print result on test
    plot_margin(X_test, Y_test, model, title='test data')


def main_people_classification():
    """
    Main function to perform people vs non people classification with SVM.
    """

    X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test = people_dataset('data')

    C = 100
    kernel = 'rbf'

    model = svm.SVC(C=C, kernel=kernel)
    model.fit(X_feat_train, Y_train)

    Y_pred = model.predict(X_feat_test)

    print('Error: {}'.format(float(np.sum(Y_pred != Y_test))/len(Y_test)))

    visualize_predictions(X_img_test, Y_test, Y_pred)


if __name__ == '__main__':
    # main_svm()
    main_people_classification()

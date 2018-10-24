import numpy as np

#from data_io import load_got_dataset, gaussians_dataset
#from logistic_regression import LogisticRegression
from lab.support_vector_machines.code import datasets
from lab.support_vector_machines.code.pegasos.SVM import SVM
from lab.support_vector_machines.code import utils
np.random.seed(191090)

def main():
    """ Main function """

    #x_train, y_train, x_test, y_test = datasets.gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    X_img_train, x_train, y_train, X_img_test, x_test, y_test = datasets.people_dataset('./data')

    svm_classifier = SVM(n_epochs=20, lam=0.001)

    # train
    svm_classifier.fit_gd(x_train, y_train, verbose=True)

    # test
    predictions = svm_classifier.predict(x_test)

    accuracy = float(np.sum(predictions == y_test)) / y_test.shape[0]
    print('Test accuracy: {}'.format(accuracy))

    if x_train.shape == 3:
        utils.plot_margin(x_train, y_train, svm_classifier)

    utils.visualize_predictions(X_img_test, y_test, predictions)

# entry point
if __name__ == '__main__':
    main()

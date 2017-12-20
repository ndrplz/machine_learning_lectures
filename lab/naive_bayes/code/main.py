"""
Main script
"""

import numpy as np
import matplotlib.pyplot as plt

from data_io import load_mnist
from naive_bayes import NaiveBayesClassifier

import itertools

plt.ion()


def plot_confusion_matrix(targets, predictions, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    n_classes, = np.unique(targets).shape

    cm = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)
    for t, p in zip(targets, predictions):
        cm[int(t), int(p)] += 1

    if normalize:
        cm /= cm.sum(axis=1)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    """ Main function """

    # load data
    x_train, y_train, x_test, y_test, label_dict = load_mnist(which_type='digits', threshold=0.5)

    # get the model
    nbc = NaiveBayesClassifier()

    # train
    nbc.fit(x_train, y_train)

    # test
    predictions = nbc.predict(x_test)

    # evaluate performances
    accuracy = np.sum(np.uint8(predictions == y_test)) / len(y_test)
    print('Accuracy: {}'.format(accuracy))

    # show confusion matrix
    plot_confusion_matrix(targets=y_test,
                          predictions=predictions,
                          classes=[label_dict[l] for l in label_dict])

    # plot predictions
    plt.figure()
    while True:
        idx = np.random.randint(0, x_test.shape[0])

        x = x_test[idx]
        p = predictions[idx]
        y = y_test[idx]

        plt.imshow(x, cmap='gray')
        plt.title('Target: {}, Prediction: {}'.format(label_dict[int(y)], label_dict[int(p)]))
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()

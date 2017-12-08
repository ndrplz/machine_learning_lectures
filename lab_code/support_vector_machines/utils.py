import numpy as np

import matplotlib.pyplot as plt
plt.ion()


def plot_margin(X, Y, model, title=''):
    """
    Represents the performance of a svm model over data.
    
    Parameters
    ----------
    X: ndarray
        data points. (shape:(n_samples, dim))
    Y: ndarray
        groundtruth labels. (shape:(n_samples,))
    model: SVC
        A sklearn.SVC fit model.
    title: str
        an optional title for the plot.
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap='jet')

    if model.kernel == 'linear':
        # get the separating hyperplane
        w = model.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(np.min(X), np.max(X))
        yy = a * xx - (model.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
        yy_down = yy + a * margin
        yy_up = yy - a * margin

        # plot the line, the points, and the nearest vectors to the plane
        ax[1].plot(xx, yy, 'k-')
        ax[1].plot(xx, yy_down, 'k--')
        ax[1].plot(xx, yy_up, 'k--')

    ax[1].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    ax[1].scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap='jet')

    plt.axis('tight')
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    # plt.figure(1, figsize=(4, 3))
    ax[1].pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.waitforbuttonpress()


def visualize_predictions(X_img, gt, pred):
    """
    Visualizes predictions of people vs non people classification.
    In an endless loop.
    
    Parameters
    ----------
    X_img: ndarray
        Images of people or non people. (shape:(n_images, h, w))
    gt: ndarray
        Groundtruth labels. (shape:(n_images,))
    pred: ndarray
        Predicted labels. (shape:(n_images,))

    Returns
    -------
    None
    """
    labels = ['non_people', 'people']

    while True:

        # sample
        idx = np.random.choice(np.arange(0, X_img.shape[0]))
        img = X_img[idx]

        plt.imshow(X_img[idx], cmap='gray')
        title = 'GT: {}, Pred: {}'.format(labels[int(gt[idx])], labels[int(pred[idx])])
        plt.title(title)

        plt.waitforbuttonpress()


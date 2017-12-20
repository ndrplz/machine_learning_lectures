import numpy as np
import matplotlib.pyplot as plt

plt.ion()

cmap = 'jet'


def plot_boundary(X, Y, model, title=''):
    """
    Represents the boundaries of a generic learning model over data.

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

    # initialize subplots
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(X[:, 0], X[:, 1], c=Y, s=40, zorder=10, cmap=cmap, edgecolors='k')

    # evaluate lims
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    # predict all over a grid
    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    ax[1].pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    ax[1].scatter(X[:, 0], X[:, 1], c=Y, s=40, zorder=10, cmap=cmap, edgecolors='k')

    # set stuff for subplots
    for s in [0, 1]:
        ax[s].set_xlim([x_min, x_max])
        ax[s].set_ylim([y_min, y_max])
        ax[s].set_xticks([])
        ax[s].set_yticks([])

    ax[0].set_title('Data')
    ax[1].set_title('Boundary')

    plt.waitforbuttonpress()

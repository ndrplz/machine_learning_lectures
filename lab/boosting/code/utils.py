import matplotlib.pyplot as plt
import numpy as np


def plot_2d_dataset(X, Y, title='', cmap='jet', blocking: bool = False):
    """
    Plots a two-dimensional dataset.

    Parameters
    ----------
    X: np.ndarray
        Data points. (shape:(n_samples, dim))
    Y: np.ndarray
        Groundtruth labels. (shape:(n_samples,))
    title: str
        Optional title for the plot.
    cmap: str
        Colormap used for plotting
    blocking: bool
        When set, wait for user interaction
    """

    plt.figure()

    # Compute and set range limits
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove ticks
    plt.xticks(())
    plt.yticks(())

    # Plot points
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, s=40, cmap=cmap, edgecolors='k')
    plt.title(title)

    if blocking:
        plt.waitforbuttonpress()


def plot_boundary(X, Y, model, title='', cmap='jet'):
    """
    Represents the boundaries of a generic learning model over data.

    Parameters
    ----------
    X: np.ndarray
        Data points. (shape:(n_samples, dim))
    Y: np.ndarray
        Ground truth labels. (shape:(n_samples,))
    model: SVC
        A sklearn classifier.
    title: str
        Optional title for the plot.
    cmap: str
        Colormap used for plotting
    """

    # Initialize subplots
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(X[:, 0], X[:, 1], c=Y, s=40, zorder=10, cmap=cmap, edgecolors='k')

    # Compute range limits
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    # Predict all over a dense grid
    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    ax[1].pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    ax[1].scatter(X[:, 0], X[:, 1], c=Y, s=40, zorder=10, cmap=cmap, edgecolors='k')

    # Set limits and ticks for each subplot
    for s in [0, 1]:
        ax[s].set_xlim([x_min, x_max])
        ax[s].set_ylim([y_min, y_max])
        ax[s].set_xticks([])
        ax[s].set_yticks([])

    ax[0].set_title(title)
    ax[1].set_title('Boundary')

    plt.waitforbuttonpress()

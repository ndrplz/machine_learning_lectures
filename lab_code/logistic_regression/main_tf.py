import numpy as np
import tensorflow as tf

from data_io import load_got_dataset, gaussians_dataset

np.random.seed(191090)


def sigmoid(X):
    return 1 / (1 + tf.exp((-X)))


def binary_crossentropy(y_true, y_pred):
    return - tf.reduce_mean(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1-y_pred))


def main():
    """ Main function """

    # x_train, y_train, x_test, y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    x_train, y_train, train_names, x_test, y_test, test_names, feature_names = load_got_dataset(path='data/got.csv', train_split=0.8)

    _, n_features = x_train.shape

    # define placeholders
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_features))
    Y = tf.placeholder(dtype=tf.float32, shape=(None,))
    w = tf.Variable(initial_value=np.random.randn(n_features), dtype=tf.float32)

    Z = sigmoid(tf.tensordot(X, w, axes=[[1], [0]]))

    loss = binary_crossentropy(y_true=Y, y_pred=Z)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)

    train_step = optimizer.minimize(loss)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for _ in range(0, 10000):

            _, loss_num = sess.run([train_step, loss], feed_dict={X: x_train, Y:y_train})

            print loss_num


# entry point
if __name__ == '__main__':
    main()

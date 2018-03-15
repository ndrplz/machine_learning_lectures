import numpy as np
import tensorflow as tf
from data import load_income_dataset


def main():

    batch_size = 64
    n_epochs = 300

    # Read data
    x_train, y_train, x_test, y_test = load_income_dataset()

    n_train_samples, sample_dim = x_train.shape
    n_train_samples, n_classes = y_train.shape
    n_test_samples, _ = x_test.shape

    # Define placeholders (1-d)
    x = tf.placeholder(shape=(None, sample_dim), dtype=tf.float32)
    y = tf.placeholder(shape=(None, n_classes), dtype=tf.float32)

    # Multi-layer perceptron
    h = tf.layers.dense(x, units=256, activation=tf.nn.leaky_relu)
    h = tf.layers.dense(h, units=128, activation=tf.nn.leaky_relu)
    h = tf.layers.dense(h, units=32, activation=tf.nn.leaky_relu)
    logits = tf.layers.dense(h, units=n_classes)
    p = tf.nn.softmax(logits)

    # Define objective function
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(p + np.finfo('float32').eps), axis=1))

    # Define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    # Define one training iteration
    train_step = optimizer.minimize(loss)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Number of batches per epoch
        batches_per_epoch = n_train_samples // batch_size

        # Train
        for i in range(n_epochs):
            total_loss = 0
            for b in range(0, batches_per_epoch):
                start = b * batch_size
                end = start + batch_size

                # Session runs train_op and fetch values of loss
                _, l, pr, log = sess.run([train_step, loss, p, logits], feed_dict={x: x_train[start:end], y: y_train[start:end]})
                total_loss += l
            print('Epoch {0}: {1}'.format(i, total_loss / batches_per_epoch))

        # Test
        test_batches = n_test_samples // batch_size
        predictions = []
        for b in range(0, test_batches + 1):
            start = b * batch_size
            end = min(start + batch_size, n_test_samples)

            # Session runs train_op and fetch values of loss
            pred, = sess.run([p], feed_dict={x: x_test[start:end], y: y_test[start:end]})
            predictions.append(pred)
        predictions = np.concatenate(predictions, axis=0)
        predictions = np.round(predictions)
        accuracy = np.sum((predictions * y_test)) / len(predictions)
        print('Test set accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    main()

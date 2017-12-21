import numpy as np
import tensorflow as tf
from utils import get_mnist_data


# MNIST classes
n_classes = 10


# This will keep model architecture definition more readable
conv2d  = tf.layers.conv2d
pool2d  = tf.layers.max_pooling2d
relu    = tf.nn.relu
dense   = tf.layers.dense
dropout = tf.nn.dropout
softmax = tf.nn.softmax


def tiny_convnet(x, keep_prob):

    with tf.variable_scope('tiny_cnn'):

        # Restore original shape for `x` (it is loaded flat)
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1_filters = 32
        h = relu(conv2d(x, filters=conv1_filters, kernel_size=[3, 3], padding='same'))
        h = pool2d(h, pool_size=[2, 2], strides=[2, 2])

        conv2_filters = 64
        h = relu(conv2d(h, filters=conv2_filters, kernel_size=[3, 3], padding='same'))
        h = pool2d(h, pool_size=[2, 2], strides=[2, 2])

        h_flat = tf.reshape(h, shape=[-1, 7*7*conv2_filters])
        h_flat = dropout(h_flat, keep_prob=keep_prob)

        probabilities = dense(h_flat, units=n_classes, activation=softmax)

        return probabilities


if __name__ == '__main__':

    # Load MNIST data
    mnist = get_mnist_data('/tmp/mnist', verbose=True)

    # Placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])     # input placeholder
    p = tf.placeholder(dtype=tf.float32)                        # dropout keep probability

    # Placeholder for targets
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # Define model output
    y = tiny_convnet(x, keep_prob=p)

    # Define loss function
    loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(y + np.finfo('float32').eps), axis=1))

    # Define train step
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init_op = tf.global_variables_initializer()

    # Define metrics
    correct_predictions = tf.equal(tf.argmax(y, axis=1), tf.argmax(targets, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(init_op)

        # Training parameters
        training_epochs = 10
        batch_size      = 128

        # Number of batches to process to see whole dataset
        batches_each_epoch = mnist.train.num_examples // batch_size

        for epoch in range(training_epochs):

            # During training measure accuracy on validation set to have an idea of what's happening
            val_accuracy = sess.run(fetches=accuracy,
                                    feed_dict={x: mnist.validation.images, targets: mnist.validation.labels, p: 1.})
            print('Epoch: {:06d} - VAL accuracy: {:.03f}'.format(epoch, val_accuracy))

            for _ in range(batches_each_epoch):

                # Load a batch of training data
                x_batch, target_batch = mnist.train.next_batch(batch_size)

                # Actually run one training step here
                sess.run(fetches=[train_step],
                         feed_dict={x: x_batch, targets: target_batch, p: 0.5})

        # Eventually evaluate on whole test set when training ends
        average_test_accuracy = 0.0
        num_test_batches = mnist.test.num_examples // batch_size
        for _ in range(num_test_batches):
            x_batch, target_batch = mnist.train.next_batch(batch_size)
            average_test_accuracy += sess.run(fetches=accuracy,
                                              feed_dict={x: x_batch, targets: target_batch, p: 1.})
        average_test_accuracy /= num_test_batches
        print('*' * 50)
        print('Training ended. TEST accuracy: {:.03f}'.format(average_test_accuracy))

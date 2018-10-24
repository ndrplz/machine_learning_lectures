import numpy as np
import tensorflow as tf
from utils import get_mnist_data


def single_layer_net(x):

    input_dim       = 784
    n_classes       = 10

    with tf.variable_scope('single_layer_net'):

        W = tf.Variable(initial_value=tf.random_normal(shape=[input_dim, n_classes]), name='weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[n_classes]), name='biases')

        logits = tf.matmul(x, W) + b

        y = tf.nn.softmax(logits)

        return y


def multi_layer_net(x):

    input_dim       = 784
    hidden_dim      = 100
    n_classes       = 10

    with tf.variable_scope('multi_layer_net'):

        W_1 = tf.Variable(initial_value=tf.random_normal(shape=[input_dim, hidden_dim]), name='l1_weights')
        b_1 = tf.Variable(initial_value=tf.zeros(shape=[hidden_dim]), name='l1_biases')

        hidden_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

        W_2 = tf.Variable(initial_value=tf.random_normal(shape=[hidden_dim, n_classes]), name='l2_weights')
        b_2 = tf.Variable(initial_value=tf.zeros(shape=[n_classes]), name='l2_biases')

        logits = tf.matmul(hidden_1, W_2) + b_2

        y = tf.nn.softmax(logits)

        return y


if __name__ == '__main__':

    # Load MNIST data
    mnist = get_mnist_data('/tmp/mnist', verbose=True)

    # Placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])     # input placeholder

    # Placeholder for targets
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # Define model output
    y = multi_layer_net(x)

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
        training_epochs = 100
        batch_size      = 128

        # Number of batches to process to see whole dataset
        batches_each_epoch = mnist.train.num_examples // batch_size

        for epoch in range(training_epochs):

            # During training measure accuracy on validation set to have an idea of what's happening
            val_accuracy = sess.run(fetches=accuracy,
                                    feed_dict={x: mnist.validation.images, targets: mnist.validation.labels})
            print('Epoch: {:06d} - VAL accuracy: {:.03f}'.format(epoch, val_accuracy))

            for _ in range(batches_each_epoch):

                # Load a batch of training data
                x_batch, target_batch = mnist.train.next_batch(batch_size)

                # Actually run one training step here
                sess.run(fetches=[train_step],
                         feed_dict={x: x_batch, targets: target_batch})

        # Eventually evaluate on whole test set when training ends
        test_accuracy = sess.run(fetches=accuracy,
                                 feed_dict={x: mnist.test.images, targets: mnist.test.labels})
        print('*' * 50)
        print('Training ended. TEST accuracy: {:.03f}'.format(test_accuracy))

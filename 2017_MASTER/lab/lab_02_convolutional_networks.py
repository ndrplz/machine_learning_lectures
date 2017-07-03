import tensorflow as tf
from lab_utils import get_mnist_data, EPS


def tiny_convnet(x, keep_prob):

    n_classes = 10

    x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.variable_scope('tiny_cnn'):

        conv1_filters = 32
        conv1 = tf.layers.conv2d(x_image, conv1_filters, kernel_size=(3, 3), padding='same')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')

        conv2_filters = 64
        conv2 = tf.layers.conv2d(pool1, conv2_filters, kernel_size=(3, 3), padding='same')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

        pool2_flat = tf.reshape(pool2, shape=(-1, 7*7*conv2_filters))

        pool2_drop = tf.nn.dropout(pool2_flat, keep_prob=keep_prob)

        hidden_units = 100
        hidden = tf.layers.dense(pool2_drop, units=hidden_units, activation=tf.nn.relu)

        logits = tf.layers.dense(hidden, units=n_classes, activation=None)

        y = tf.nn.softmax(logits)

        return y


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
    loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(y + EPS), reduction_indices=1))

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
        test_accuracy = sess.run(fetches=accuracy,
                                 feed_dict={x: mnist.test.images, targets: mnist.test.labels, p: 1.})
        print('*' * 50)
        print('Training ended. TEST accuracy: {:.03f}'.format(test_accuracy))













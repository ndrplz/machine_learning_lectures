import tensorflow as tf
from lab_utils import get_mnist_data, EPS


def my_little_pony_net(x):

    with tf.variable_scope('my_little_pony_net'):

        # todo Compute a more reasonable prediction here
        # todo Prediction must be a distribution over 10 MNIST classes,
        # todo  that is to say, a vector of ten numbers which sum to 1.
        y = 0

        return y


if __name__ == '__main__':

    # Load MNIST data
    mnist = get_mnist_data('/tmp/mnist', verbose=True)

    # Placeholders
    # todo x =

    # Placeholder for targets
    # todo targets =

    # Define model output
    # todo y = my_little_pony_net(x)

    # Define loss function (i.e. categorical cross-entropy)
    # todo loss =

    # Define train step
    # todo train_step =

    init_op = tf.global_variables_initializer()

    # Define metrics (e.g. accuracy in percentage)
    # todo accuracy = #

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(init_op)

        # # Training parameters
        # todo training_epochs =
        # todo batch_size      =

        # # Number of batches to process to see whole dataset
        # batches_each_epoch = mnist.train.num_examples // batch_size
        #
        # for epoch in range(training_epochs):
        #
        #     # During training measure accuracy on validation set to have an idea of what's happening
        #     val_accuracy = sess.run(fetches=accuracy,
        #                             feed_dict={x: mnist.validation.images, targets: mnist.validation.labels})
        #     print('Epoch: {:06d} - VAL accuracy: {:.03f}'.format(epoch, val_accuracy))
        #
        #     for _ in range(batches_each_epoch):
        #
        #         # Load a batch of training data
        #         x_batch, target_batch = mnist.train.next_batch(batch_size)
        #
        #         # Actually run one training step here
        #         sess.run(fetches=[train_step],
        #                  feed_dict={x: x_batch, targets: target_batch})
        #
        # # Eventually evaluate on whole test set when training ends
        # test_accuracy = sess.run(fetches=accuracy,
        #                          feed_dict={x: mnist.test.images, targets: mnist.test.labels})
        # print('*' * 50)
        # print('Training ended. TEST accuracy: {:.03f}'.format(test_accuracy))

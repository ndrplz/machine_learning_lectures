import tensorflow as tf
from lab_utils import get_mnist_data, EPS


def tiny_convnet(x, keep_prob):

    # todo Tip: depending on your definition of `x` placeholder, you may have to reshape `x`
    # todo Tip: the output `y` must be a distribution over the ten possible classes.
    # todo Tip: start small. A couple of conv+relu+pool stacks may already lead to good results.
    with tf.variable_scope('tiny_cnn'):
        y = 0
        return y


if __name__ == '__main__':

    # Load MNIST data
    mnist = get_mnist_data('/tmp/mnist', verbose=True)

    # Placeholders
    # todo x =      # input placeholder
    # todo p =      # dropout keep probability

    # Placeholder for targets
    # todo targets =

    # Define model output
    # todo y = tiny_convnet(x, keep_prob=p)

    # Define loss function (e.g. categorical crossentropy)
    # loss

    # Define train step
    # train_step

    init_op = tf.global_variables_initializer()

    # Define metrics
    # todo accuracy =

    # with tf.Session() as sess:
    #
    #     # Initialize all variables
    #     sess.run(init_op)
    #
    #     # Training parameters
    #     # todo training_epochs =
    #     # todo batch_size      =
    #
    #     # Number of batches to process to see whole dataset
    #     batches_each_epoch = mnist.train.num_examples // batch_size
    #
    #     for epoch in range(training_epochs):
    #
    #         # During training measure accuracy on validation set to have an idea of what's happening
    #         val_accuracy = sess.run(fetches=accuracy,
    #                                 feed_dict={x: mnist.validation.images, targets: mnist.validation.labels, p: 1.})
    #         print('Epoch: {:06d} - VAL accuracy: {:.03f}'.format(epoch, val_accuracy))
    #
    #         for _ in range(batches_each_epoch):
    #
    #             # Load a batch of training data
    #             x_batch, target_batch = mnist.train.next_batch(batch_size)
    #
    #             # Actually run one training step here
    #             sess.run(fetches=[train_step],
    #                      feed_dict={x: x_batch, targets: target_batch, p: 0.5})
    #
    #     # Eventually evaluate on whole test set when training ends
    #     average_test_accuracy = 0.0
    #     num_test_batches = mnist.test.num_examples // batch_size
    #     for _ in range(num_test_batches):
    #         x_batch, target_batch = mnist.train.next_batch(batch_size)
    #         average_test_accuracy += sess.run(fetches=accuracy,
    #                                           feed_dict={x: x_batch, targets: target_batch, p: 1.})
    #     average_test_accuracy /= num_test_batches
    #     print('*' * 50)
    #     print('Training ended. TEST accuracy: {:.03f}'.format(average_test_accuracy))

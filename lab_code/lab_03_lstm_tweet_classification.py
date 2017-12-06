import tensorflow as tf
import numpy as np
from lab_utils import load_tweet_batch, EPS


if __name__ == '__main__':

    max_seq_len         = 20
    max_dict_size       = 1000
    train_tweets_path   = 'data/tweets_train.csv'
    val_tweets_path     = 'data/tweets_val.csv'

    # Training parameters
    training_epochs     = 1000
    batch_size          = 32
    batches_each_epoch  = 50

    x = tf.placeholder(dtype=tf.float32, shape=[None, max_seq_len, max_dict_size + 1])
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    hidden_size = 5                                  # LSTM cell dimension
    state = tf.zeros([batch_size, hidden_size])        # Initial state of the LSTM memory.

    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

    val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    val = tf.transpose(val, [1, 0, 2])              # [batchsize, seq_len, dict_len] -> [seq_len, batchsize, dict_len]
    last = tf.gather(val, max_seq_len - 1)          # Select final output only

    W = tf.Variable(tf.truncated_normal([hidden_size, int(targets.get_shape()[1])]))
    b = tf.Variable(tf.constant(0.0, shape=[targets.get_shape()[1]]))

    predictions = tf.sigmoid(tf.matmul(last, W) + b)

    loss = -tf.reduce_sum(targets * tf.log(predictions + EPS) + (1.0 - targets) * (tf.log(1. - predictions + EPS)))

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Accuracy metrics
    mistakes = tf.equal(tf.round(predictions), targets)
    accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    with tf.Session() as sess:

        # Initialize all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for epoch in range(training_epochs):

            x_batch, y_batch = load_tweet_batch(train_tweets_path, 1024, max_seq_len, max_dict_size)
            print('TRAIN  - Loss: {:.02f}     Accuracy: {:.02f}'.format(
                sess.run(loss, {x: x_batch, targets: np.expand_dims(y_batch, 1)}),
                sess.run(accuracy, {x: x_batch, targets: np.expand_dims(y_batch, 1)})
            ))

            x_batch, y_batch = load_tweet_batch(val_tweets_path, 1024, max_seq_len, max_dict_size)
            print('VAL  - Loss: {:.02f}     Accuracy: {:.02f}'.format(
                sess.run(loss, {x: x_batch, targets: np.expand_dims(y_batch, 1)}),
                sess.run(accuracy, {x: x_batch, targets: np.expand_dims(y_batch, 1)})
            ))

            for _ in range(batches_each_epoch):

                # Load a batch of training data
                x_batch, y_batch = load_tweet_batch(train_tweets_path, batch_size, max_seq_len, max_dict_size)

                # Actually run one training step here
                sess.run(fetches=[train_step],
                         feed_dict={x: x_batch, targets: np.expand_dims(y_batch, 1)})

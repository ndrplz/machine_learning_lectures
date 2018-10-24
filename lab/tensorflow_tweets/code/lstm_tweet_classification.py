import numpy as np
import tensorflow as tf
from utils import TweetLoader, EPS


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


class TweetModel(object):

    def __init__(self, x, targets, hidden_size):

        self.x = x
        self.targets = targets
        self.n_classes = targets.get_shape()[-1]

        self.hidden_size = hidden_size

        self.inference = None
        self.loss = None
        self.train_step = None
        self.accuracy = None

        self.make_inference()
        self.make_loss()
        self.make_train_step()
        self.make_accuracy()

    def make_inference(self):

        # Create LSTM cell with proper hidden size
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)

        # Get LSTM output
        val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=length(x))

        # Get last output of LSTM
        last = last_relevant(val, length(val))

        # Define the final prediction applying a fully connected layer with softmax
        self.inference = tf.layers.dense(inputs=last, units=self.n_classes, activation=tf.nn.softmax)

    def make_loss(self):
        self.loss = - tf.reduce_sum(targets * tf.log(self.inference + EPS))

    def make_train_step(self):
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def make_accuracy(self):
        mistakes = tf.equal(tf.argmax(self.inference, axis=1), tf.argmax(self.targets, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))


if __name__ == '__main__':

    max_seq_len = 20
    max_dict_size = 1000
    hidden_size = 10                                   # LSTM cell dimension
    train_tweets_path = 'data/tweets_train.csv'
    val_tweets_path = 'data/tweets_val.csv'

    # Training parameters
    training_epochs = 20
    batch_size = 32
    batches_each_epoch = 500

    # Get tweet loader
    loader = TweetLoader(train_tweets_path, val_tweets_path, batch_size, max_seq_len, max_dict_size)

    # Declare placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, max_seq_len, max_dict_size + 1])
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    # Get a model
    model = TweetModel(x, targets, hidden_size)

    # Open new session
    sess = tf.Session()

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        x_batch, y_batch = loader.load_tweet_batch(mode='train')
        print('Epoch: {}\tTRAIN: Loss: {:.02f} Accuracy: {:.02f}'.format(
            epoch,
            sess.run(model.loss, {x: x_batch, targets: y_batch}),
            sess.run(model.accuracy, {x: x_batch, targets: y_batch})
        ))

        x_batch, y_batch = loader.load_tweet_batch(mode='val')
        print('Epoch: {}\tVAL: Loss: {:.02f} Accuracy: {:.02f}'.format(
            epoch,
            sess.run(model.loss, {x: x_batch, targets: y_batch}),
            sess.run(model.accuracy, {x: x_batch, targets: y_batch})
        ))

        for _ in range(batches_each_epoch):

            # Load a batch of training data
            x_batch, y_batch = loader.load_tweet_batch(mode='train')

            # Actually run one training step here
            sess.run(fetches=[model.train_step],
                     feed_dict={x: x_batch, targets: y_batch})

    # Interactive session
    while True:
        tw = raw_input('Try tweeting something!')
        if tw:
            x_num = loader.vectorize(tweet=tw)
            p, = sess.run([model.inference], feed_dict={x: x_num})
            if np.argmax(p) == 0:
                # Negative tweet
                print('Prediction:{}\t:('.format(p))
            else:
                print('Prediction:{}\t:)'.format(p))
        else:
            break

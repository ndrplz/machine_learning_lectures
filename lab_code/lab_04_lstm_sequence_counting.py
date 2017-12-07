import argparse
import numpy as np
import tensorflow as tf
from synthetic_dataset import SyntheticSequenceDataset


class DeepCounter:

    def __init__(self, x, targets, args):

        self.x = x
        self.targets   = targets
        self.n_classes = targets.get_shape()[-1]

        self.hidden_size = args.hidden_size
        self.epsilon     = args.eps

        self._inference  = None
        self._loss       = None
        self._train_step = None
        self._accuracy   = None

        self.inference
        self.loss
        self.train_step
        self.accuracy

    @property
    def inference(self):
        if self._inference is None:

            # Create LSTM cell
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)

            # Define the recurrent network
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs=self.x, dtype=tf.float32)

            # Take the last output in the sequence
            last_output = outputs[:, -1, :]

            # Final dense layer to get to the prediction
            self._inference = tf.layers.dense(last_output, units=self.n_classes, activation=tf.nn.softmax)

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            self._loss = - tf.reduce_sum(targets * tf.log(self.inference + self.epsilon))
        return self._loss

    @property
    def train_step(self):
        if self._train_step is None:
            self._train_step = tf.train.AdamOptimizer().minimize(self.loss)
        return self._train_step

    @property
    def accuracy(self):
        if self._accuracy is None:
            correct_predictions = tf.equal(tf.round(self.inference), targets)
            self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))
        return self._accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size of LSTM', metavar='')
    parser.add_argument('--batch_size', type=np.int32, default=128, help='Batch size', metavar='')
    parser.add_argument('--training_epochs', type=np.int32, default=3, help='Number of training epochs', metavar='')
    parser.add_argument('--eps', type=np.float32, default=np.finfo('float32').eps,  help='Machine epsilon', metavar='')
    args = parser.parse_args()

    synthetic_dataset   = SyntheticSequenceDataset()

    # Define placeholders
    data    = tf.placeholder(dtype=tf.float32, shape=[None, 20, 1])
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 21])

    deep_counter = DeepCounter(x=data, targets=targets, args=args)

    with tf.Session() as sess:

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Load dataset
        train_data, train_targets, test_data, test_targets = synthetic_dataset.data

        # Training parameters
        training_epochs    = args.training_epochs
        batch_size         = args.batch_size
        batches_each_epoch = int(len(train_data)) // batch_size

        print('\n' + 50*'*' + '\nTraining\n' + 50*'*')

        # Train batch by batch
        for epoch in range(0, training_epochs):

            start_idx = 0
            loss_current_epoch = []
            for _ in range(0, batches_each_epoch):

                # Load batch
                end_idx = start_idx + batch_size
                data_batch, target_batch = train_data[start_idx:end_idx], train_targets[start_idx:end_idx]

                # Run one optimization step on current step
                cur_loss, _ = sess.run(fetches=[deep_counter.loss, deep_counter.train_step],
                                       feed_dict={data: data_batch, targets: target_batch})
                loss_current_epoch.append(cur_loss)

                # Update data pointer
                start_idx += batch_size

            print('Epoch {:02d} - Loss on train set: {:.02f}'.format(epoch, sum(loss_current_epoch)/batches_each_epoch))

        print('\n' + 50 * '*' + '\nTesting\n' + 50 * '*')

        accuracy_score = 0.0
        num_test_batches = int(len(test_data)) // batch_size
        start_idx = 0

        # Test batch by batch
        for _ in range(0, num_test_batches):
            end_idx = start_idx + batch_size
            data_batch, target_batch = test_data[start_idx:end_idx], test_targets[start_idx:end_idx]
            accuracy_score += sess.run(deep_counter.accuracy, {data: data_batch, targets: target_batch})
            start_idx += batch_size

        print('Average accuracy on test set: {:.03f}'.format(accuracy_score / num_test_batches))

        print('\n' + 50 * '*' + '\nInteractive Session\n' + 50 * '*')

        while True:
            my_sequence = input('Write your own binary sequence 20 digits in {0, 1}:\n')
            if my_sequence:

                # Pad shorter sequences
                if len(my_sequence) < 20:
                    my_sequence = (20 - len(my_sequence))*'0' + my_sequence

                # Crop longer sequences
                my_sequence = my_sequence[:20]

                # Prepare example
                test_example = []
                for binary_char in my_sequence:
                    test_example.append([float(binary_char)])

                pred = sess.run(deep_counter.inference, feed_dict={data: np.expand_dims(test_example, 0)})
                print('Predicted number of ones: {} - Real: {}\n'.format(int(np.argmax(pred)), int(np.sum(test_example))))
            else:
                break

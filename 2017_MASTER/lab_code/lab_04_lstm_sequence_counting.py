import argparse
import tensorflow as tf
from synthetic_dataset import SyntheticSequenceDataset
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size of LSTM', metavar='')
    parser.add_argument('--batch_size', type=np.int32, default=128, help='Batch size', metavar='')
    parser.add_argument('--training_epochs', type=np.int32, default=3, help='Number of training epochs', metavar='')
    parser.add_argument('--eps', type=np.float32, default=np.finfo('float32').eps,  help='Machine epsilon', metavar='')
    args = parser.parse_args()

    n_classes           = 21  # there are 21 = 0, 1, ..., 20 different classes
    synthetic_dataset   = SyntheticSequenceDataset()

    # Define placeholders
    data    = tf.placeholder(dtype=tf.float32, shape=[None, 20, 1])
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 21])

    # Create LSTM cell
    hidden_size = args.hidden_size
    cell        = tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)

    # Define the recurrent network
    val, _  = tf.nn.dynamic_rnn(cell, inputs=data, dtype=tf.float32)

    # Take the last output in the sequence
    val = tf.transpose(val, perm=[1, 0, 2])
    last_output = tf.gather(val, val.get_shape()[0]-1)

    # Final dense layer to get to the prediction
    W = tf.get_variable(name='weights', shape=[hidden_size, n_classes], dtype=tf.float32)
    b = tf.get_variable(name='biases', shape=[n_classes], dtype=tf.float32)
    prediction = tf.nn.softmax(tf.matmul(last_output, W) + b)

    # Loss function is the usual categorical cross-entropy
    cross_entropy = - tf.reduce_sum(targets * tf.log(prediction + args.eps))

    # Define training step
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # Accuracy metrics
    correct_predictions = tf.equal(tf.round(prediction), targets)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))

    with tf.Session() as sess:

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Load dataset
        train_data, train_targets, test_data, test_targets = synthetic_dataset.data

        # Training parameters
        training_epochs     = args.training_epochs
        batch_size          = args.batch_size
        batches_each_epoch  = int(len(train_data)) // batch_size

        print('\n' + 50*'*' + '\nTraining\n' + 50*'*')

        # Train batch by batch
        for epoch in range(0, training_epochs):
            print('Epoch {:02d}'.format(epoch))
            start_idx = 0
            for _ in range(0, batches_each_epoch):

                # Load batch
                end_idx     = start_idx + batch_size
                data_batch, target_batch = train_data[start_idx:end_idx], train_targets[start_idx:end_idx]

                # Run one optimization step on current step
                sess.run(train_step, {data: data_batch, targets: target_batch})

                # Update data pointer
                start_idx += batch_size

        print('\n' + 50 * '*' + '\nTesting\n' + 50 * '*')

        accuracy_score = 0.0
        num_test_batches = int(len(test_data)) // batch_size
        start_idx = 0

        # Test batch by batch
        for _ in range(0, num_test_batches):
            end_idx = start_idx + batch_size
            data_batch, target_batch = test_data[start_idx:end_idx], test_targets[start_idx:end_idx]
            accuracy_score += sess.run(accuracy, {data: data_batch, targets: target_batch})
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

                pred = sess.run(prediction, feed_dict={data: np.expand_dims(test_example, 0)})
                print('Predicted number of ones: {} - Real: {}\n'.format(int(np.argmax(pred)), int(np.sum(test_example))))
            else:
                break

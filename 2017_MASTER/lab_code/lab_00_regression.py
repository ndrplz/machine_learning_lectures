import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def read_data(csv_file):
    body_weight  = []
    brain_weight = []
    with open(csv_file, 'rt') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            if row:
                idx, brain_w, body_w = row[0].split()
                brain_weight.append(float(brain_w))
                body_weight.append(float(body_w))

    return body_weight, brain_weight


if __name__ == '__main__':

    plt.ion()

    body_weight, brain_weight = read_data('data/brain_body_weight.txt')
    n_samples = len(body_weight)

    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)

    w = tf.Variable(initial_value=0.0)
    b = tf.Variable(initial_value=0.0)

    y_pred = x * w + b

    loss = tf.reduce_mean(tf.square(y - y_pred))

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    train_step = optimizer.minimize(loss)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(100):  # train the model 100 times
            total_loss = 0
            for bo_w, br_w in zip(body_weight, brain_weight):
                # Session runs train_op and fetch values of loss
                _, l = sess.run([train_step, loss], feed_dict={x: bo_w, y: br_w})
                total_loss += l
            print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

            w_value, b_value = sess.run([w, b])

            # plot the results
            plt.cla()
            plt.plot(body_weight, brain_weight, 'bo', label='Real data')
            plt.plot(np.arange(0, int(max(body_weight))), np.arange(0, int(max(body_weight))) * w_value + b_value, 'r', label='Predicted data')
            plt.pause(0.5)

        plt.legend()





import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import get_brain_body_data
from google_drive_downloader import GoogleDriveDownloader


if __name__ == '__main__':

    plt.ion()   # interactive mode

    # Download tiles data
    GoogleDriveDownloader.download_file_from_google_drive(file_id='1EOYhZJxdOBi81qICtmYpqKYzDlbb_Pmm',
                                                          dest_path='./brain_body_weight.zip',
                                                          overwrite=True,
                                                          unzip=True)

    # Read data
    body_weight, brain_weight = get_brain_body_data('./brain_body_weight.txt')
    n_samples = len(body_weight)

    # Define placeholders (1-d)
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)

    # Define variables
    w = tf.Variable(initial_value=0.0)
    b = tf.Variable(initial_value=0.0)

    # Linear regression model
    y_pred = x * w + b

    # Define objective function
    loss = tf.square(y - y_pred)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    # Define one training iteration
    train_step = optimizer.minimize(loss)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            total_loss = 0
            for bo_w, br_w in zip(body_weight, brain_weight):
                # Session runs train_op and fetch values of loss
                _, l = sess.run([train_step, loss], feed_dict={x: bo_w, y: br_w})
                total_loss += l
            print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

            # Plot current results
            plt.cla()  # clear axis first
            w_value, b_value = sess.run([w, b])  # get current numeric solutions
            plt.plot(body_weight, brain_weight, 'bo', label='Real data')
            plt.plot(np.arange(0, int(max(body_weight))),
                     np.arange(0, int(max(body_weight))) * w_value + b_value,
                     'r', label='Predicted data')
            plt.pause(0.5)

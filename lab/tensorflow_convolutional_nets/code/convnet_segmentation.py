import tensorflow as tf
import numpy as np
from time import time
from google_drive_downloader import GoogleDriveDownloader
from utils import TilesDataset
from utils import convert_target_to_one_hot


# This will keep model architecture definition more readable
conv2d  = tf.layers.conv2d
pool2d  = tf.layers.max_pooling2d
relu    = tf.nn.relu
dense   = tf.layers.dense
dropout = tf.nn.dropout
softmax = tf.nn.softmax


class TileSegmenter:

    def __init__(self, x, targets):

        self.x       = x
        self.targets = targets

        self._inference     = None
        self._loss          = None
        self._train_step    = None
        self._summaries     = None

        self.inference
        self.loss
        self.train_step
        self.summaries

    @property
    def inference(self):
        if self._inference is None:
            with tf.variable_scope('inference'):
                conv1 = conv2d(self.x, filters=32, kernel_size=[3, 3], padding='same', activation=relu)
                pool1 = pool2d(conv1, pool_size=[2, 2], strides=[2, 2], padding='same')

                conv2 = conv2d(pool1, filters=64, kernel_size=[3, 3], padding='same', activation=relu)
                pool2 = pool2d(conv2, pool_size=[2, 2], strides=[2, 2], padding='same')

                conv3_a = conv2d(pool2,   filters=64, kernel_size=[3, 3], padding='same', activation=relu)
                conv3_b = conv2d(conv3_a, filters=64, kernel_size=[3, 3], padding='same', activation=relu)
                pool3   = pool2d(conv3_b, pool_size=[2, 2], strides=[2, 2], padding='same')

                conv4_a = conv2d(pool3,   filters=128, kernel_size=[3, 3], padding='same', activation=relu)
                conv4_b = conv2d(conv4_a, filters=128, kernel_size=[3, 3], padding='same', activation=relu)

                conv_final = conv2d(conv4_b, 2, kernel_size=(1, 1), padding='same')

                self._inference = tf.image.resize_bilinear(conv_final, size=(64, 64))

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            with tf.variable_scope('loss'):
                # Define loss function
                targets_flat    = tf.reshape(self.targets, shape=(-1, 2))
                prediction_flat = tf.reshape(self.inference, shape=(-1, 2))
                cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=targets_flat, logits=prediction_flat)
                self._loss  = tf.reduce_sum(cross_entropies)
        return self._loss

    @property
    def train_step(self):
        if self._train_step is None:
            with tf.variable_scope('training'):
                self._train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
        return self._train_step

    @property
    def summaries(self):
        if self._summaries is None:
            # Add TensorBoard Summaries
            how_many_images = 6

            # --- scalar summaries
            tf.summary.scalar('loss', self.loss)
            tf.summary.image('input', self.x, max_outputs=6)

            # --- background image summaries
            bg_target_image         = tf.expand_dims(tf.gather(tf.transpose(self.targets, [3, 0, 1, 2]), 0), axis=3)
            bg_pred_image           = tf.expand_dims(tf.gather(tf.transpose(self.inference, [3, 0, 1, 2]), 0), axis=3)
            bg_pred_image_rounded   = tf.round(tf.nn.softmax(self.inference, dim=3))
            bg_pred_image_rounded   = tf.expand_dims(tf.gather(tf.transpose(bg_pred_image_rounded, [3, 0, 1, 2]), 0), axis=3)

            tf.summary.image('BACKGROUND (targets)',            bg_target_image,        max_outputs=how_many_images)
            tf.summary.image('BACKGROUND (prediction)',         bg_pred_image,          max_outputs=how_many_images)
            tf.summary.image('BACKGROUND ROUNDED (prediction)', bg_pred_image_rounded,  max_outputs=how_many_images)

            # --- foreground image summaries
            fg_target_image         = tf.expand_dims(tf.gather(tf.transpose(self.targets, [3, 0, 1, 2]), 1), axis=3)
            fg_pred_image           = tf.expand_dims(tf.gather(tf.transpose(self.inference, [3, 0, 1, 2]), 1), axis=3)
            fg_pred_image_rounded   = tf.round(tf.nn.softmax(self.inference, dim=3))
            fg_pred_image_rounded   = tf.expand_dims(tf.gather(tf.transpose(fg_pred_image_rounded, [3, 0, 1, 2]), 1), axis=3)

            tf.summary.image('FOREGROUND (targets)',            fg_target_image,        max_outputs=how_many_images)
            tf.summary.image('FOREGROUND (prediction)',         fg_pred_image,          max_outputs=how_many_images)
            tf.summary.image('FOREGROUND ROUNDED (prediction)', fg_pred_image_rounded,  max_outputs=how_many_images)

            # --- merge all summaries and initialize the summary writer
            self._summaries = tf.summary.merge_all()
        return self._summaries


if __name__ == '__main__':

    # Download tiles data
    GoogleDriveDownloader.download_file_from_google_drive(file_id='1W58D4qVZtUAFprDdoC9KRyBWT0k0ie5r',
                                                          dest_path='./tiles.zip',
                                                          overwrite=True,
                                                          unzip=True)
    # Load tiles dataset
    tiles_dataset = TilesDataset(dataset_root='./toy_dataset_tiles')

    # Placeholders
    x       = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])         # input
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 2])         # target

    with tf.Session() as sess:

        # Define model output
        model = TileSegmenter(x, targets)

        # FileWriter to save Tensorboard summary
        train_writer = tf.summary.FileWriter('checkpoints/{}'.format(time()), graph=sess.graph)

        # Training parameters
        training_epochs = 1000
        batch_size = 128

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Number of batches to process to see whole dataset
        batches_each_epoch = tiles_dataset.train_num_examples // batch_size

        for epoch in range(training_epochs):

            epoch_loss = 0.0

            idx_start = 0
            for _ in range(batches_each_epoch):

                idx_end = idx_start + batch_size

                # Load a batch of training data
                x_batch         = np.array(tiles_dataset.train_x[idx_start:idx_end])
                target_batch    = np.array(tiles_dataset.train_y[idx_start:idx_end])

                # Convert the target batch into one-hot encoding (from 64x64x1 to 64x64x2)
                target_batch = convert_target_to_one_hot(target_batch)

                # Preprocess train batch
                x_batch -= 128.0

                # Actually run one training step here
                _, cur_loss = sess.run(fetches=[model.train_step, model.loss],
                                       feed_dict={x: x_batch, targets: target_batch})

                idx_start = idx_end

                epoch_loss += cur_loss

            summaries = sess.run(model.summaries, feed_dict={x: x_batch, targets: target_batch})
            train_writer.add_summary(summaries, epoch)

            print('Epoch: {:03d} - Loss: {:.02f}'.format(epoch, epoch_loss / batches_each_epoch))

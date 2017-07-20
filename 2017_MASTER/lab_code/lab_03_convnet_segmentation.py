import tensorflow as tf
import numpy as np
from lab_utils import load_tiles_dataset_from_cache
from lab_utils import TilesDataset


class TileSegmenter:

    def __init__(self, x, targets):

        self.x      = x
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
                conv1 = tf.layers.conv2d(self.x, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')

                conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

                conv3_a = tf.layers.conv2d(pool2,   filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
                conv3_b = tf.layers.conv2d(conv3_a, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(conv3_b, pool_size=(2, 2), strides=(2, 2), padding='same')

                conv4_a = tf.layers.conv2d(pool3,   filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
                conv4_b = tf.layers.conv2d(conv4_a, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

                conv_final = tf.layers.conv2d(conv4_b, 2, kernel_size=(1, 1), padding='same')

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
            with tf.variable_scope('summaries'):
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

    # Load TILES data
    tiles_dataset = load_tiles_dataset_from_cache('data/tiles_dataset.pickle')

    # Placeholders
    x       = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])         # input
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 2])         # target

    with tf.Session() as sess:

        # Define model output
        model = TileSegmenter(x, targets)

        # FileWriter to save Tensorboard summary
        train_writer = tf.summary.FileWriter('checkpoints', graph=sess.graph)

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

                # Each example in the target batch is a tensor of shape 64x64x1
                # This must be converted into one-hot tensor of shape 64x64x2
                def convert_target_to_one_hot(target_batch):
                    b, h, w, c = target_batch.shape
                    out_tensor = np.zeros(shape=(b, h, w, 2))
                    for k, cur_example in enumerate(target_batch):
                        foreground_mask = np.squeeze(cur_example > 0)
                        background_mask = np.squeeze(cur_example == 0)
                        out_tensor[k, background_mask, 0] = 1.0
                        out_tensor[k, foreground_mask, 1] = 1.0
                    return out_tensor

                # Convert the target batch into one-hot encoding
                target_batch = convert_target_to_one_hot(target_batch)

                # Preprocess train batch
                x_batch -= 128.0

                # Actually run one training step here
                _, cur_loss = sess.run(fetches=[model.train_step, model.loss], feed_dict={x: x_batch, targets: target_batch})

                idx_start = idx_end

                epoch_loss += cur_loss

            summaries = sess.run(model.summaries, feed_dict={x: x_batch, targets: target_batch})
            train_writer.add_summary(summaries, epoch)

            print('Epoch: {:03d} - Loss: {:.02f}'.format(epoch, epoch_loss / batches_each_epoch))

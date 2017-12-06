import tensorflow as tf
import numpy as np
from lab_utils import load_tiles_dataset_from_cache
from time import time
from lab_utils import TilesDataset


# This will keep model architecture definition more readable
conv2d  = tf.layers.conv2d
pool2d  = tf.layers.max_pooling2d
relu    = tf.nn.relu
dense   = tf.layers.dense
dropout = tf.nn.dropout
softmax = tf.nn.softmax


def convert_target_to_one_hot(target_batch):
    """
    Convert a batch of targets from 64x64x1 to 64x64x2 one-hot encoding.
    """
    b, h, w, c = target_batch.shape
    out_tensor = np.zeros(shape=(b, h, w, 2))
    for k, cur_example in enumerate(target_batch):
        foreground_mask = np.squeeze(cur_example > 0)
        background_mask = np.squeeze(cur_example == 0)
        out_tensor[k, background_mask, 0] = 1.0
        out_tensor[k, foreground_mask, 1] = 1.0
    return out_tensor


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

                # todo Implement a reasonable forward pass
                # todo Tip: the output will be a batch of predicted grayscale masks,
                # todo  i.e. it will have shape [batchsize, 64, 64, 1]
                self._inference = None

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            with tf.variable_scope('loss'):
                # todo Define the loss function here
                # todo Tip: you will likely need `tf.reshape` to unroll both targets and predictions,
                # todo  remember that these are available in `self.targets` and `self.inference`
                # todo Tip: why re-inventing the wheel? Check `tf.nn.softmax_cross_entropy_with_logits`
                self._loss  = None
        return self._loss

    @property
    def train_step(self):
        if self._train_step is None:
            with tf.variable_scope('training'):
                # todo Define training step here
                self._train_step = None
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

    # Load TILES data
    tiles_dataset = load_tiles_dataset_from_cache('data/tiles_protocol_02.pickle')

    # Placeholders
    x = None        # todo input placeholder
    targets = None  # todo target placeholder

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

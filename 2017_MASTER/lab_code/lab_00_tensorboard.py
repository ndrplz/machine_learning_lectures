import tensorflow as tf

if __name__ == '__main__':

    x = tf.placeholder(tf.float32)

    y = tf.sin(x)
    tf.add_summary( )
    while True:

__all__ = ['random_slice', 'cast_type', 'add_depth_dim']


import tensorflow as tf


def random_slice(X, y, training=True, max_offset=0, axis=1):
    """
    Slice a tensor X along axis, beginning at a random offset up to max_offset,
    taking (X.shape[axis] - max_offset) samples.
    If training==False, this will take the last N-max_offset samples.
    """
    if training:
        offset = tf.random.uniform(shape=[], minval=0, maxval=max_offset, dtype=tf.int32)
    else:
        offset = max_offset
    n_subsamps = X.shape[axis] - max_offset
    if axis == 0:
        if len(y.shape) > axis and y.shape[axis] == X.shape[axis]:
            y = tf.slice(y, [offset, 0], [n_subsamps, -1])
        X = tf.slice(X, [offset, 0], [n_subsamps, -1])
    else:  # axis == 1
        if len(y.shape) > axis and y.shape[axis] == X.shape[axis]:
            y = tf.slice(y, [0, offset], [-1, n_subsamps])
        X = tf.slice(X, [0, offset], [-1, n_subsamps])
    return X, y


def cast_type(x_dat, y_dat, x_type=tf.float32, y_type=tf.uint8):
    x_dat = tf.cast(x_dat, x_type)
    y_dat = tf.cast(y_dat, y_type)
    return x_dat, y_dat


def add_depth_dim(x_dat, y_dat):
    x_dat = tf.expand_dims(x_dat, -1)  # Prepare as an image, with only 1 colour-depth channel.
    return x_dat, y_dat

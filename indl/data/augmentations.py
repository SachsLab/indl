__all__ = ['random_slice', 'cast_type', 'add_depth_dim']


import tensorflow as tf


def random_slice(X, y, training=True, max_offset=0, axis=1):
    """
    Slice a tensor X along axis, beginning at a random offset up to max_offset,
    taking (X.shape[axis] - max_offset) samples.
    If training==False, this will take the last N-max_offset samples.
    Args:
        X (tf.tensor): input tensor
        y (tf.tensor): input labels
        training (bool): if the model is run in training state
        max_offset (int): number of samples
        axis (int): axis along which to slice

    Returns:
        tf.tensor, tf.tensor: X, y tuple randomly sliced.
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


def cast_type(X, y, x_type=tf.float32, y_type=tf.uint8):
    """
    Cast input pair to new dtypes.
    Args:
        X (tf.tensor): Input tensor
        y (tf.tensor): Input labels
        x_type (tf.dtypes): tf data type
        y_type (tf.dtypes): tf data type

    Returns:
        tf.tensor, tf.tensor: X, y tuple, each cast to its new type.
    """
    x_dat = tf.cast(X, x_type)
    y_dat = tf.cast(y, y_type)
    return x_dat, y_dat


def add_depth_dim(X, y):
    """
    Add extra dimension at tail for x only. This is trivial to do in-line.
    This is slightly more convenient than writing a labmda.
    Args:
        X (tf.tensor):
        y (tf.tensor):

    Returns:
        tf.tensor, tf.tensor: X, y tuple, with X having a new trailing dimension.

    """
    x_dat = tf.expand_dims(X, -1)  # Prepare as an image, with only 1 colour-depth channel.
    return x_dat, y

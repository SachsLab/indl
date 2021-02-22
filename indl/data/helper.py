__all__ = ['get_tf_dataset']


import tensorflow as tf
from functools import partial
from .augmentations import *


def get_tf_dataset(X, Y, training=True, batch_size=8, max_offset=0, slice_ax=1):
    """
    """
    # TODO: trn_test as arg

    if isinstance(training, tuple):
        ds_train = get_tf_dataset(X[trn_test[0]], Y[trn_test[0]], training=True, batch_size=batch_size)
        ds_test = get_tf_dataset(X[trn_test[1]], Y[trn_test[1]], training=False, batch_size=batch_size)
        return ds_train, ds_test

    _ds = tf.data.Dataset.from_tensor_slices((X, Y))

    _ds = _ds.map(add_depth_dim)

    _ds = _ds.map(cast_type)

    slice_fun = partial(random_slice, training=training, max_offset=max_offset, axis=slice_ax)
    _ds = _ds.map(slice_fun)

    if training:
        _ds = _ds.shuffle()

    _ds = _ds.batch(X.shape[0] + 1, drop_remainder=not training)

    return _ds

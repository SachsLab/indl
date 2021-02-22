__all__ = ['get_tf_dataset']


import tensorflow as tf
from functools import partial
from .augmentations import *


def get_tf_dataset(X, Y, training=True, batch_size=8, max_offset=0, slice_ax=1):
    """
    Convert a pair of tf tensors into a tf.data.Dataset with some augmentations.
    The added augmentations are:

    - `add_depth_dim` (with default params)
    - `cast_type` (with default params)
    - `random_slice`
    Args:
        X (tf.tensor): X data - must be compatible with above augmentations.
        Y (tf.tensor): Y data - must be compatible with above augmentations.
        training (bool or tuple): passed to `random_slice`, or if a tuple
            (e.g. from sklearn.model_selection.train_test_split) then this function returns training and test sets.
        batch_size (int): Unused I think.
        max_offset (int): Passed to `random_slice`
        slice_ax (int): Passed to `random_slice`

    Returns:
        tf.data.Dataset(, tf.Dataset): A tensorflow dataset with extra augmentations. If training is a tuple
         then two datasets are returning: training set and test set.
    """
    # TODO: trn_test as arg

    if isinstance(training, tuple):
        ds_train = get_tf_dataset(X[training[0]], Y[training[0]], training=True, batch_size=batch_size)
        ds_test = get_tf_dataset(X[training[1]], Y[training[1]], training=False, batch_size=batch_size)
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

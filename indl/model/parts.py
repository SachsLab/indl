__all__ = ['DepthwiseConv2DBlock', 'SeparableConv2DBlock', 'CombineInputs', 'EEGNetEncPartA', 'EEGNetEncPartB',
           'EEGNetEncPartC', 'EEGNetEnc', 'Bottleneck', 'Classify', 'UndoBottleneck', 'EEGNetDecodeA', 'EEGNetDecodeB']


import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, constraints
from ..regularizers import KernelLengthRegularizer
from .helper import check_inputs


@check_inputs
def DepthwiseConv2DBlock(_input,
                         depth_multiplier=4,
                         depth_pooling=1,
                         dropout_rate=0.25, dropout_type='Dropout',
                         activation='elu',
                         return_model=None):
    if dropout_type == 'SpatialDropout2D':
        dropout_type = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    _y = layers.DepthwiseConv2D((_input.shape.as_list()[0], 1), use_bias=False,
                                depth_multiplier=depth_multiplier,
                                depthwise_constraint=constraints.max_norm(1.))(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)
    if depth_pooling > 1:
        _y = layers.AveragePooling2D((1, depth_pooling))(_y)
    _y = dropout_type(dropout_rate)(_y)

    if return_model is False:
        return _y
    else:
        return models.Model(inputs=_input, outputs=_y)


def SeparableConv2DBlock(_input,
                    n_kerns=4, kern_space=1, kern_length=64,
                    kern_regu_scale=0.0, l1=0.000, l2=0.000,
                    n_pool=4,
                    dropout_rate=0.25, dropout_type='Dropout',
                    activation='elu',
                    return_model=None):

    if dropout_type == 'SpatialDropout2D':
        dropout_type = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    num_chans = _input.shape.as_list()[0]

    _y = layers.SeparableConv2D(n_kerns, (kern_space, kern_length),
                                padding='same', use_bias=False)(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)
    if n_pool > 1:
        _y = layers.AveragePooling2D((1, n_pool))(_y)
    _y = dropout_type(dropout_rate)(_y)

    if return_model is False:
        return _y
    else:
        return models.Model(inputs=_input, outputs=_y)


def CombineInputs(input_shapes):
    inputs, subblocks = [], []
    for shape in input_shapes:
        inputs.append(layers.Input(shape=shape))
        subblocks.append(inputs)

    concat = layers.Concatenate(axis=-1, name='concat')(subblocks)
    return models.Model(inputs=inputs, outputs=concat)


@check_inputs
def EEGNetEncPartA(_input,
                   F1=4, F1_kernLength=64,
                   F1_kern_reg=None,
                   l1_reg=0.000, l2_reg=0.000,
                   return_model=None):
    if F1_kern_reg is None and (l1_reg > 0 or l2_reg > 0):
        F1_kern_reg = tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    # Temporal-filter-like.
    # Applies F1 different filters along time domain (same F1 filters applied to each channel).
    # (num_chans, num_samples, 1) --> (num_chans, num_samples, F1)
    _y = layers.Conv2D(F1, (1, F1_kernLength), padding='same',
                            kernel_regularizer=F1_kern_reg,
                            # use_bias=False
                            )(_input)
    _y = layers.BatchNormalization()(_y)
    if return_model is False:
        return _y
    else:
        return models.Model(inputs=_input, outputs=_y)


@check_inputs
def EEGNetEncPartB(_input,
                   D=2,
                   D_pooling=4,
                   dropoutRate=0.25,
                   activation='elu',
                   l1_reg=0.000, l2_reg=0.000,
                   return_model=None):
    # Spatial-filter-like.
    # Applies D different filters along channels domain for each F1 (temporal filter above).
    # As our channels-domain filter is the number of channels, and we use padding='valid',
    # the channels domain is reduced from n_channels to 1.
    # (num_chans, num_samples, F1) --> (1, num_samples, F1*D)
    depth_regu = tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    _y = layers.DepthwiseConv2D((_input.shape.as_list()[1], 1),
                                padding='valid',
                                # use_bias=False,
                                depth_multiplier=D,
                                depthwise_regularizer=depth_regu,
                                depthwise_constraint=constraints.max_norm(1.))(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)

    # Smooth in the time-dimension
    # (1, num_samples, F1 * D) --> (1, num_samples // D_pooling, F1 * D)
    _y = layers.AveragePooling2D((1, D_pooling))(_y)
    _y = layers.Dropout(dropoutRate)(_y)

    if return_model is False:
        return _y
    else:
        return models.Model(inputs=_input, outputs=_y)


@check_inputs
def EEGNetEncPartC(_input,
                   F2=8,
                   F2_kernLength=16,
                   F2_pooling=8,
                   dropoutRate=0.25,
                   activation='elu',
                   l1_reg=0.000, l2_reg=0.000,
                   return_model=None):

    # Aggregate features in the time-dimension.
    # Applies F1*D time-domain filters of length F2_kernLength.
    # Then applies F2 different 1x1x(F1*D) pointwise filters.
    # (1, num_samples // D_pooling, F1 * D) --> (1, num_samples // D_pooling, F2)
    depth_regu = tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    _y = layers.SeparableConv2D(F2, (1, F2_kernLength), padding='same',
                                depthwise_regularizer=depth_regu,
                                # use_bias=False
                                )(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)

    # Further time-domain smoothing
    # (1, num_samples // D_pooling, F2) --> (1, num_samples // D_pooling // F2_pooling, F2)
    _y = layers.AveragePooling2D((1, F2_pooling))(_y)
    _y = layers.Dropout(dropoutRate)(_y)

    if return_model is False:
        return _y
    else:
        return models.Model(inputs=_input, outputs=_y)


@check_inputs
def EEGNetEnc(_input,
              F1=4, F1_kernLength=64,
              F1_kern_reg=None,
              D=2, D_pooling=4,
              F2=8, F2_kernLength=16,
              F2_pooling=8,
              dropoutRate=0.25,
              activation='elu',
              l1_reg=0.000, l2_reg=0.000,
              return_model=None):
    """
    Model builder function. This function is called to construct the encoder part of EEGNet.
    Args:
        input: a single input. Can be one of:
            - a single shape tuple (with batch dim in axis=0)
            - a dict with 1 key for input name and corresponding value is shape tuple
            - a single tf.TensorSpec
        dropoutRate: dropout fraction
        F1_kernLength: length of temporal convolution in first layer. We found
            that setting this to be half the sampling rate worked
            well in practice.
        F1: number of temporal filters
        F2: number of pointwise filters
        D: number of spatial filters to learn within each temporal convolution.
        dropoutType: Either SpatialDropout2D or Dropout, passed as a string.
        norm_rate: penalty for max norm on classifiers
    Returns:
        model: A tensorflow model object
    """
    if isinstance(input, list):
        raise ValueError("This model builder only accepts a single input.")

    # This model block requires that the input have shape (batch, channels, samples, filters).
    input_shape = _input.shape.as_list()
    # Usually the filter dimension does not exist so let's add that.
    if len(input_shape) < 4:
        input_shape = input_shape + [1]
    # The Conv layers are insensitive to the number of samples in the time dimension.
    # To make it possible for this trained model to be applied to segments of different
    # durations, we need to explicitly state that we don't care about the number of samples.
    # input_shape[2] = -1  # Comment out during debug
    _y = layers.Reshape(input_shape[1:])(_input)  # Note that Reshape ignores the batch dimension.

    # Temporal filter-alike
    _y = EEGNetEncPartA(_y,
                        F1=F1, F1_kernLength=F1_kernLength,
                        F1_kern_reg=F1_kern_reg,
                        l1_reg=l1_reg, l2_reg=l2_reg,
                        return_model=False)

    # Spatial-filter-alike.
    _y = EEGNetEncPartB(_y,
                        D=D,
                        D_pooling=D_pooling,
                        dropoutRate=dropoutRate,
                        activation=activation,
                        l1_reg=l1_reg, l2_reg=l2_reg,
                        return_model=False)

    # Aggregate features in the time-dimension.
    _y = EEGNetEncPartC(_y,
                        F2=F2,
                        F2_kernLength=F2_kernLength,
                        F2_pooling=F2_pooling,
                        dropoutRate=dropoutRate,
                        activation=activation,
                        l1_reg=l1_reg, l2_reg=l2_reg,
                        return_model=False)

    if return_model is False:
        return _y
    else:
        return models.Model(inputs=_input, outputs=_y)


@check_inputs
def Bottleneck(_input, latent_dim=32, activation='elu', dropoutRate=0.25,
               return_model=None):

    y = layers.Flatten()(_input)
    # Force through bottleneck
    if latent_dim:
        y = layers.Dense(latent_dim)(y)
        if dropoutRate > 0:
            y = layers.Dropout(dropoutRate)(y)
    if activation is not None:
        y = layers.Activation(activation)(y)

    if return_model is False:
        return y
    else:
        return models.Model(inputs=_input, outputs=y)


@check_inputs
def Classify(_input, n_classes=1, norm_rate=0.25, return_model=None):
    output = layers.Dense(n_classes, name='classifier',
                          activation='sigmoid' if n_classes==1 else 'softmax',
                          kernel_constraint=constraints.max_norm(norm_rate))(_input)
    if return_model is False:
        return output
    else:
        return models.Model(inputs=input, outputs=output)


@check_inputs
def UndoBottleneck(_input, target_shape=(1, 1, 8), dropoutRate=0.25, activation='elu', return_model=None):
    _x = layers.Dense(tf.reduce_prod(target_shape).numpy())(_input)
    _x = layers.Activation(activation)(_x)
    _x = layers.BatchNormalization(axis=-1)(_x)
    _x = layers.Dropout(dropoutRate)(_x)
    output = layers.Reshape(target_shape=target_shape)(_x)  # F2 * unpack, --> 1, n_unpack, F2

    if return_model is False:
        return output
    else:
        return tf.keras.Model(inputs=_input, outputs=output)


@check_inputs
def EEGNetDecodeA(_input,
                  F1=4, D=2, F2_kernLength=16, F2_pooling=8,
                  activation='elu', dropoutRate=0.25,
                  l1_reg=0.000, l2_reg=0.000,
                  return_model=None):
    input_shape = _input.shape.as_list()[1:]
    # input_shape[1] = -1
    _x = layers.Reshape(input_shape)(_input)

    # Undo pointwise (second) part of final SeparableConv2D.
    _x = layers.SeparableConv2D(F1 * D, (1, 1), padding='same',
                                depthwise_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                )(_x)

    # Up-sample as first part of undoing smoothing
    _x = layers.UpSampling2D(size=(1, F2_pooling))(_x)
    # Undo first part of final SeparableConv2D. This helps undo smoothing.
    _x = layers.DepthwiseConv2D((1, F2_kernLength), padding='same', depth_multiplier=1, use_bias=False,
                                depthwise_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                )(_x)
    _x = layers.Activation(activation)(_x)
    _x = layers.BatchNormalization(axis=-1)(_x)
    output = layers.Dropout(dropoutRate)(_x)

    if return_model is False:
        return output
    else:
        return tf.keras.Model(inputs=_input, outputs=output)


@check_inputs
def EEGNetDecodeB(_input, num_chans=1, D_pooling=4,
                  F1=4, F1_kernLength=64,
                  activation='elu', dropoutRate=0.25,
                  l1_reg=0.000, l2_reg=0.000,
                  return_model=None):
    # Doesn't care about time-domain so let's make that -1 (--> None)
    input_shape = _input.shape.as_list()[1:]
    # input_shape[1] = -1
    _x = layers.Reshape(input_shape)(_input)

    # Undo spatial filter
    # (1, num_samples // D_pooling, F1 * D) --> (num_chans, num_samples // D_pooling, F1 * D)
    _x = layers.UpSampling2D(size=(num_chans, 1))(_x)
    _x = layers.DepthwiseConv2D(kernel_size=(num_chans, 1), padding='same', use_bias=False,
                                depthwise_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                )(_x)
    _x = layers.Activation(activation)(_x)
    _x = layers.BatchNormalization(axis=-1)(_x)
    _x = layers.Dropout(dropoutRate)(_x)

    # Undo smoothing that happens after spatial filter.
    # (num_chans, num_samples // D_pooling, F1 * D) --> (num_chans, num_samples // D_pooling * D_pooling, F1)
    _x = layers.UpSampling2D(size=(1, D_pooling))(_x)
    _x = layers.SeparableConv2D(F1, (1, F1_kernLength), padding='same', use_bias=False,
                                depthwise_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                )(_x)
    _x = layers.Activation(activation)(_x)
    _x = layers.BatchNormalization(axis=-1)(_x)
    output = layers.Dropout(dropoutRate)(_x)

    if return_model is False:
        return output
    else:
        return tf.keras.Model(inputs=_input, outputs=output)


@check_inputs
def Conv2DBlockRegu(_input,
                    n_kerns=4, kern_space=1, kern_length=64,
                    kern_regu_scale=0.0, l1=0.000, l2=0.000,
                    n_pool=4,
                    dropout_rate=0.25, dropout_type='Dropout',
                    activation='elu',
                    return_model=None):
    if dropout_type == 'SpatialDropout2D':
        dropout_type = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    num_chans = _input.shape.as_list()[0]

    if kern_regu_scale:
        kern_regu = KernelLengthRegularizer((1, kern_length),
                                            window_func='poly',
                                            window_scale=kern_regu_scale,
                                            poly_exp=2, threshold=0.0015)
    elif l1 > 0 or l2 > 0:
        kern_regu = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
    else:
        kern_regu = None

    # Temporal-filter-like
    _y = layers.Conv2D(n_kerns, (kern_space, kern_length),
                       padding='same',
                       kernel_regularizer=kern_regu,
                       use_bias=False)(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)
    if n_pool > 1:
        _y = layers.AveragePooling2D((1, n_pool))(_y)
    _y = dropout_type(dropout_rate)(_y)

    if return_model is False:
        return _y
    else:
        return models.Model(inputs=input, outputs=_y)


@check_inputs
def UNEyeBlock(_input, num_filts=10, filt_shape=(1, 3),
               padding='same',
               kern_reg=None,
               return_model=None):
    """
    UNeye is a UNet for gaze classification. Its encoder uses a series of conv-relu-batchnorm blocks
    https://github.com/berenslab/uneye/blob/master/uneye/functions.py
    :param input:
    :param num_filts:
    :param filt_shape:
    :param return_model:
    :return:
    """
    _y = layers.Conv2D(num_filts, filt_shape, padding=padding, kernel_regularizer=kern_reg)(_input)
    _y = layers.Activation('relu')(_y)
    _y = layers.BatchNormalization()(_y)

    if return_model is False:
        return _y
    else:
        return tf.keras.Model(inputs=_input, outputs=_y)

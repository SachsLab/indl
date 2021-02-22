__all__ = ['normalize', 'upsample_timeseries', 'get_maximizing_inputs', 'visualize_layer']


import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import scipy.interpolate


"""
#Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes.

Results example: ![Visualization](http://i.imgur.com/4nj4KjN.jpg)
"""


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def upsample_timeseries(test_dat, n_resamples, axis=1):
    if test_dat.shape[axis] == n_resamples:
        return test_dat

    y = test_dat.numpy()
    x = np.linspace(0, 1, y.shape[axis])
    f_interp = scipy.interpolate.interp1d(x, y, axis=axis)
    xnew = np.linspace(0, 1, n_resamples)
    y = f_interp(xnew)
    return tf.convert_to_tensor(y.astype(np.float32))


def _stitch_filters(max_acts, n=None, sort_by_activation=True):
    """Draw the best filters in a nxn grid.

    # Arguments
        filters: A List of generated images and their corresponding losses
                 for each processed filter.
        n: dimension of the grid.
           If none, the largest possible square will be used
    """
    if n is None:
        n = int(np.floor(np.sqrt(len(max_acts))))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top n*n filters.
    if sort_by_activation:
        max_acts.sort(key=lambda x: x[1], reverse=True)
    max_acts = max_acts[:n * n]

    output_dim = max_acts[0][0].shape

    act_dat = np.stack([_[0] for _ in max_acts], axis=1)
    for act_ix in range(act_dat.shape[1]):
        temp = act_dat[:, act_ix, :]
        f_min, f_max = temp.min(), temp.max()
        act_dat[:, act_ix, :] = (temp - f_min) / (f_max - f_min)

    MARGIN = 5
    n_x = n * output_dim[0] + (n - 1) * MARGIN
    stitched_filters = np.nan * np.ones((n_x, n, output_dim[1]))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            filt_ix = i * n + j
            if filt_ix < act_dat.shape[1]:
                dat = act_dat[:, filt_ix, :]
                width_margin = (output_dim[0] + MARGIN) * i
                stitched_filters[width_margin: width_margin + output_dim[0], j, :] = dat - j

    return stitched_filters


def get_maximizing_inputs(model, layer_ix, max_filts=None, n_steps=100, at_others_expense=False):
    in_shape = [1] + model.input.shape.as_list()[1:]

    layer_output = model.layers[layer_ix].output
    n_filts = layer_output.shape[-1]

    filt_ids = np.arange(n_filts)
    if max_filts is not None:
        if isinstance(max_filts, (list, tuple)):
            filt_ids = np.array(max_filts)
        elif n_filts > max_filts:
            filt_ids = np.random.permutation(n_filts)[:max_filts]

    filt_slice = [np.s_[:] for _ in range(K.ndim(layer_output))]

    input_tups = []
    for ix, filt_ix in enumerate(filt_ids):
        input_data = tf.convert_to_tensor(np.random.randn(*in_shape).astype(np.float32))
        trunc_model = tf.keras.Model(model.input, layer_output)
        filt_slice[-1] = filt_ix
        non_targ_id = tf.constant(np.setdiff1d(np.arange(layer_output.shape[-1], dtype=int), filt_ix))
        loss_value = 0
        for step_ix in range(n_steps):
            with tf.GradientTape() as tape:
                tape.watch(input_data)
                layer_act = trunc_model(input_data)
                filt_act = layer_act[filt_slice]
                if at_others_expense:
                    nontarg_act = K.mean(tf.gather(layer_act, non_targ_id, axis=-1))
                    loss_value = K.mean(filt_act - nontarg_act)
                else:
                    loss_value = K.mean(filt_act)
            grads = tape.gradient(loss_value, input_data)  # Derivative of loss w.r.t. input
            # Normalize gradients
            grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
            input_data += grads
        input_data = np.squeeze(input_data)
        input_tups.append((filt_ix, input_data, loss_value.numpy()))

    return input_tups


def visualize_layer(model, layer_idx,
                    loss_as_exclusive=False,
                    output_dim=(701, 58), filter_range=(0, None),
                    step=1., epochs=200,
                    upsampling_steps=9, upsampling_factor=1.2
                    ):
    """Visualizes the most relevant filters of one conv-layer in a certain model.
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
    # Arguments
        model: The model containing layer_name.
        layer_idx: The index of the layer to be visualized.
        loss_as_exclusive: If True, loss also minimizes activations of non-test filters in the layer.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upsampling_steps: Number of upscaling steps. Currently not working.
        upsampling_factor: Factor to which to slowly upgrade the timeseries towards output_dim. Currently not working.
        output_dim: [n_timesteps, n_channels] The output image dimensions.
        filter_range: [lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`, the last filter will be inferred as the upper boundary.
    """

    output_layer = model.layers[layer_idx]

    max_filts = len(output_layer.get_weights()[1])
    max_filts = max_filts if filter_range[1] is None else min(max_filts, filter_range[1])

    # iterate through each filter in this layer and generate its activation-maximization time series
    maximizing_activations = []
    for f_ix in range(filter_range[0], max_filts):
        s_time = time.time()
        if loss_as_exclusive:
            model_output = output_layer.output
        else:
            if isinstance(output_layer, tf.keras.layers.Conv1D):
                model_output = output_layer.output[:, :, f_ix]
            else:
                model_output = output_layer.output[:, f_ix]
        max_model = tf.keras.Model(model.input, model_output)

        # we start with some random noise that is smaller than the expected output.
        n_samples_out = output_dim[0]
        n_samples_intermediate = int(n_samples_out / (upsampling_factor ** upsampling_steps))
        test_dat = tf.convert_to_tensor(
            np.random.random((1, n_samples_intermediate, output_dim[-1])).astype(np.float32))

        for up in reversed(range(upsampling_steps)):
            # Run gradient ascent
            for _ in range(epochs):
                with tf.GradientTape() as tape:
                    tape.watch(test_dat)
                    layer_act = max_model(test_dat)
                    if not loss_as_exclusive:
                        loss_value = K.mean(layer_act)
                    else:
                        from_logits = output_layer.activation != tf.keras.activations.softmax
                        loss_value = K.sparse_categorical_crossentropy(f_ix, K.mean(layer_act, axis=-2),
                                                                       from_logits=from_logits)[0]

                gradients = tape.gradient(loss_value, test_dat)
                # normalization trick: we normalize the gradient
                gradients = normalize(gradients)
                test_dat += gradients * step

                # some filters get stuck to 0, re-init with random data.
                # These will probably end up being low-loss activations.
                if loss_value <= K.epsilon():
                    test_dat = tf.convert_to_tensor(
                        np.random.random((1, n_samples_intermediate, output_dim[-1])).astype(np.float32))

            # Now upsample the timeseries
            n_samples_intermediate = int(n_samples_intermediate / (upsampling_factor ** up))
            test_dat = upsample_timeseries(test_dat, n_samples_intermediate, axis=1)

        print('Costs of filter: {:5.0f} ( {:4.2f}s )'.format(loss_value.numpy(), time.time() - s_time))
        test_dat = upsample_timeseries(test_dat, n_samples_out, axis=1)
        maximizing_activations.append((test_dat[0].numpy(), loss_value.numpy()))

    print('{} filters processed.'.format(len(maximizing_activations)))
    return maximizing_activations

__all__ = ['GenerativeRNN']


import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.util import nest
from tensorflow.python.keras.layers.recurrent import _standardize_args
import tensorflow.keras.layers as tfkl


class GenerativeRNN(tfkl.RNN):
    """Generative RNN.
    This is a wrapper around the normal RNN layer, except that it does not require
    an input. If an input is given, a time dimension will be added if not provided,
    and if the provided time dimension has length less than `timesteps`, it will be
    tile-padded with the last sample (or with zeros if `tile_input=False`).
    If an input is not given, zeros input will be assumed; in that case the batch
    size may come from initial_state if provided. If neither the input nor the
    initial_state is provided then the batch size cannot be known, but the output
    would always be zeros anyway so this RNN is quite useless.
    If mask is provided, it is used to choose the last input sample from which the
    remaining input is tile-padded (or zero-padded). The output will not be masked.
    -Warning: mask has not been thoroughly tested.

    Arguments:
        cell: recurrent cell instance instance. See tfkl.RNN for description.
        timesteps: integer number of timesteps to generate.
        See tfkl.RNN for other kwargs.
    """

    def __init__(self,
                 cell,
                 timesteps=1,
                 tile_input=False,  # If True, it will tile the last input, else it will pad with zeros
                 **kwargs):
        super().__init__(cell, **kwargs)
        self.timesteps = timesteps
        self.tile_input = tile_input
        self._input_has_time_dim = False
        self._batch_dims: int = kwargs.pop('batch_dims', 1)
        self._output_spec = None
        self._built_with_input = False

    def _fixup_input_shape(self, input_shape):
        if input_shape is None:
            # We will make a fake input with feature length = 1
            input_shape = (None, 1)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        # Check for a time dimension
        time_ax_ix = 0 if self.time_major else -2
        if len(input_shape) < 3 or input_shape[time_ax_ix] is None:
            # No time dimension provided. Add one.
            if self.time_major:
                input_shape = (self.timesteps,) + input_shape
            else:
                input_shape = input_shape[:-1] + (self.timesteps, input_shape[-1])

        # Pretend that the time dimension has self.timesteps so that
        # the output gets calculated correctly.
        if input_shape[time_ax_ix] != self.timesteps:
            if self.time_major:
                input_shape = (self.timesteps,) + input_shape[1:]
            else:
                input_shape = input_shape[:-2] + (self.timesteps, input_shape[-1])
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['timesteps'] = self.timesteps
        return config

    def build_with_input(self, inputs, *args, **kwargs):
        bd = self._batch_dims
        # self._input_spec = [tf.nest.map_structure(
        #     lambda x: tfkl.InputSpec(shape=[None] * bd + x.shape[bd:], dtype=x.dtype), inputs)]
        dummy_input = tf.nest.map_structure(lambda t: tf.zeros([2] * bd + t.shape[bd:], t.dtype), inputs)
        dummy_output = super().__call__(dummy_input, *args, **kwargs)
        self._output_spec = tf.nest.map_structure(lambda x: tfkl.InputSpec(shape=[None] * bd + x.shape[bd:],
                                                                           dtype=x.dtype), dummy_output)
        self._built_with_input = True

    @property
    def output_spec(self):
        return self._output_spec

    @output_spec.setter
    def output_spec(self, value):
        self._output_spec = value

    @property
    def output_shape(self):
        assert self.output_spec is not None, 'build_with_input has not been called; output shape is not defined'
        return tf.nest.map_structure(lambda x: x.shape, self.output_spec)

    @property
    def output_dtype(self):
        assert self.output_spec is not None, 'build_with_input has not been called; output dtype is not defined'
        return tf.nest.map_structure(lambda x: x.dtype, self.output_spec)

    # @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape=None):
        input_shape = self._fixup_input_shape(input_shape)
        if self.output_spec is None:
            return super().compute_output_shape(input_shape)
        batch_shape = tf.nest.flatten(input_shape)[0][:self.batch_dims]
        return tf.nest.map_structure(lambda x: batch_shape + x[self.batch_dims:], self.output_shape)

    def build(self, input_shape=None):
        input_shape = self._fixup_input_shape(input_shape)
        super().build(input_shape)

    def __call__(self, *args, inputs=None, initial_state=None, constants=None, mask=None, **kwargs):

        inputs, initial_state, constants = _standardize_args(inputs,
                                                             initial_state,
                                                             constants,
                                                             self._num_constants)
        # We allow different shapes of input, even None. It doesn't really matter
        # because ultimately the input will be ignored except for the first step.
        # Nevertheless, we expand the input to have a timesteps dimension. This
        # is done simply for parent class calculations of output size, etc.

        # Allow None as an input. We will create an array of zeros of appropriate shape.
        if inputs is None:
            if initial_state is not None:
                # If LSTM then state might be a list.
                _state = initial_state[0] if isinstance(initial_state, list) else initial_state
                batch_size = _state.shape[:-1]
                inputs = K.zeros_like(_state[..., 0][..., tf.newaxis])
                # inputs = 0 * _state[..., 0][..., tf.newaxis]  # Assume dim=1 input
            else:
                # Neither inputs nor initial_state provided. This likely only happens
                # when building/testing the layer.
                inputs = tf.zeros((self.timesteps, 1, 1)) if self.time_major else tf.zeros((1, self.timesteps, 1))

        # Allow 2D input, here reshape to 3D input
        if len(K.int_shape(inputs)) < 3:
            if self.time_major:
                inputs = inputs[tf.newaxis, ...]
            else:
                inputs = inputs[..., tf.newaxis, :]

        time_ax_ix, batch_ax_ix = (0, 1) if self.time_major else (-2, 0)
        input_shape = K.int_shape(inputs)
        input_timesteps = input_shape[time_ax_ix]

        if mask is not None and K.any(~mask):
            mask = nest.flatten(mask)[0]
            # We assume mask has a time dimension and require it is same size as input
            # (It doesn't make sense to use mask otherwise).
            mask_shape = K.int_shape(mask)
            # If the mask only has 1 item in the batch dim then tile it
            if mask_shape[batch_ax_ix] == 1 and input_shape[batch_ax_ix] > 1:
                if self.time_major:
                    bcast_or = tf.zeros((1, input_shape[batch_ax_ix], 1), dtype=tf.bool)
                else:
                    bcast_or = tf.zeros((input_shape[batch_ax_ix], 1, 1), dtype=tf.bool)
                mask = tf.math.logical_or(mask, bcast_or)
            if mask_shape[time_ax_ix] == input_timesteps:
                # Prepare slice parameters
                # For head (kept)
                h_sl_begin = [0 for _ in input_shape]
                h_sl_sz = [-1 for _ in input_shape]
                h_sl_sz[batch_ax_ix] = 1
                # For tail (replaced)
                t_sl_begin = [0 for _ in input_shape]
                t_sl_sz = [-1 for _ in input_shape]
                t_sl_sz[batch_ax_ix] = 1
                # Collect input replacements in list
                new_inputs = []
                for batch_ix in range(input_shape[batch_ax_ix]):
                    samp_mask = mask[..., batch_ix, :] if self.time_major else mask[batch_ix]
                    if K.any(~samp_mask):
                        h_sl_begin[batch_ax_ix] = batch_ix
                        t_sl_begin[batch_ax_ix] = batch_ix
                        first_bad = tf.where(~samp_mask)[0, 0]
                        h_sl_sz[time_ax_ix] = first_bad  # sz is 1-based
                        t_sl_begin[time_ax_ix] = first_bad
                        head = tf.slice(inputs, h_sl_begin, h_sl_sz)
                        tail = tf.slice(inputs, t_sl_begin, t_sl_sz)
                        if self.tile_input:
                            tile_samp = head[-1] if self.time_major else head[..., -1, :]
                        else:
                            tile_samp = tf.zeros((1, input_shape[-1]))
                        new_row = tf.concat((head, tile_samp * K.ones_like(tail)), axis=time_ax_ix)
                        new_inputs.append(new_row)
                inputs = tf.concat(new_inputs, axis=batch_ax_ix)

        # Fill/trim input time dimension to be self.timesteps
        if input_timesteps > self.timesteps:
            # Trim excess, if any
            inputs = inputs[:self.timesteps, ...] if self.time_major else inputs[..., :self.timesteps, :]
        elif input_timesteps < self.timesteps:
            # Take the last timestep as our starting point for the padding data
            pad_sample = inputs[-1] if self.time_major else inputs[..., -1, :]
            if not self.tile_input:
                # zero out padding data if we aren't tiling
                pad_sample = K.zeros_like(pad_sample)
                # pad_sample = 0 * pad_sample
            # Add the time axis back to our pad_sample
            pad_sample = pad_sample[tf.newaxis, ...] if self.time_major else pad_sample[..., tf.newaxis, :]
            # How many more timestamps do we need?
            pad_timestamps = self.timesteps - K.int_shape(inputs)[time_ax_ix]
            # Tile pad_data using broadcast-add. Does this same line work for time_major and not?
            pad_data = pad_sample + tf.zeros((pad_timestamps, 1))
            inputs = tf.concat((inputs, pad_data), axis=time_ax_ix)

        if not self._built_with_input:
            self.build_with_input(inputs, *args, initial_state=initial_state, constants=constants, mask=mask, **kwargs)

        return super().__call__(inputs, initial_state=initial_state,
                                constants=constants, **kwargs)

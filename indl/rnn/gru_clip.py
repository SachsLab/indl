import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.recurrent import GRU as GRUv1
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.util import nest


class GRUClipCell(tfkl.GRUCell):
    # A couple differences between tfkl GRUCell and LFADS CustomGRU
    # * different variable names (this:LFADS)
    #    - z:u; r:r; h:candidate
    # * stacking order. tfkl stacks z,r,h all in one : LFADS stacks r,u in '_gate' and c in '_candidate'
    # * tfkl recurrent_activation is param and defaults to hard_sigmoid : LFADS is always sigmoid
    def __init__(self, units, clip_value=np.inf, init_gate_bias_ones=True, **kwargs):
        super(GRUClipCell, self).__init__(units, **kwargs)
        self._clip_value = clip_value
        self._init_gate_bias_ones = init_gate_bias_ones

    def build(self, input_shape):
        super(GRUClipCell, self).build(input_shape)
        # * tfkl initializes all bias as zeros by default : LFADS inits gate's to ones and candidate's to zeros
        # * tfkl has separate input_bias and recurrent_bias : LFADS has recurrent_bias only
        if self._init_gate_bias_ones:
            init_weights = self.get_weights()
            if not self.reset_after:
                init_weights[2][:2*self.units] = 1.
            else:
                # separate biases for input and recurrent. We only modify recurrent.
                init_weights[2][1][:2 * self.units] = 1.
            self.set_weights(init_weights)

    def call(self, inputs, states, training=None):
        h, _ = super().call(inputs, states, training=training)
        h = tf.clip_by_value(h, -self._clip_value, self._clip_value)
        new_state = [h] if nest.is_sequence(states) else h
        return h, new_state


class GRUClip(GRUv1):
    # Note: Doesn't work with GRUv2 because it has some GPU optimizations that aren't compatible here.
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 clip_value=np.inf, init_gate_bias_ones=True,
                 **kwargs):
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                           kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = GRUClipCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            reset_after=reset_after,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            clip_value=clip_value,
            init_gate_bias_ones=init_gate_bias_ones,
            **cell_kwargs
        )
        super(GRUv1, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

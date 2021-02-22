__all__ = ['ComplexCell', 'test_ComplexCell']


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from .utils import DiagonalGaussianFromExisting, GRUClipCell


class ComplexCell(tfkl.AbstractRNNCell):
    _BIAS_VARIABLE_NAME = "bias"
    _WEIGHTS_VARIABLE_NAME = "kernel"
    """Cell class for the LFADS Generative GRU + Controller Input

    This cell uses two GRUClipCells: One for the Generator and one for the Controller.
    The Generator   - inputs: the output of the Controller cell and optionally 'external' inputs.
                    - initial state: in LFADS -- a sample of a posterior distribution that is parameterized
                      by an encoder.
    The Controller  - inputs: the concatenation of the encoded controller inputs and the generator cell's state
                      from the previous iteration transformed through the factor dense layer.
                    - initial state: in LFADS -- a learnable Variable of zeros.

    The two cells share the same initialization parameters
    (activations, initializers, bias, dropout, regularizer, etc.) except for the number of units.

    Arguments:
        units_gen: Positive integer, number of units in generator RNN cell.
        units_con: Positive integer, number of units in controller RNN cell.
        factors_dim: Number of units in Dense layer for factors output.
            This layer would normally be external to the RNN. However, in LFADS, the factors dense layer
            is also used to transform the t-1 generator cell state which becomes part of the _inputs_
            to the controller cell.
        co_dim: Dimensionality of variational posterior from controller output --> inputs to controller RNN
        ext_input_dim: Size of external input. The cell input will be split into encoded_z and ext_input depending
            on this value. Can be 0.
        inject_ext_input_to_gen: Only makes sense if ext_input_dim is > 0, and `False` is not implemented.
        activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass None, no activation is applied (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
          Default: hard sigmoid (`hard_sigmoid`).
          If you pass `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
          Note: LFADS uses normal sigmoid.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
          Default: lecun_normal
          Vanilla tensorflow default is glorot_uniform.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
          used for the linear transformation of the recurrent state.
          Default: orthogonal
          LFADS uses lecun_normal
        bias_initializer: Initializer for the bias vector.
          Default: zeros
          Note: LFADS uses ones for gate bias and zeros for candidate bias
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
          Default: None
        recurrent_regularizer: Regularizer function applied to the `recurrent_kernel` weights matrix.
          Default: 'l2' at 0.01
          Note: LFADS uses L2 regularization with per-cell scaling.
          Default for generator is 2000., and for controller is 0. (sum(v*v)*scale*0.5) / numel
        bias_regularizer: Regularizer function applied to the bias vector.
          Default: None
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
          Default: None
        recurrent_constraint: Constraint function applied to the `recurrent_kernel` weights matrix.
          Default: None
        bias_constraint: Constraint function applied to the bias vector.
          Default: None
        dropout: Float between 0 and 1.
          Fraction of the units to drop for the linear transformation of the inputs.
          Default: 0.05
        recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for the linear transformation of the recurrent state.
          Default: 0.0
        implementation: Implementation mode, either 1 or 2.
          Mode 1 will structure its operations as a larger number of
          smaller dot products and additions, whereas mode 2 will
          batch them into fewer, larger operations. These modes will
          have different performance profiles on different hardware and
          for different applications.
          Note: This applies to the sub-cells.
        reset_after: GRU convention (whether to apply reset gate after or
          before matrix multiplication). False = "before" (default),
          True = "after" (CuDNN compatible).
        clip_value: Value at which to clip the GRU cell output.
          Default: np.inf (no clipping)

    Call arguments:
        inputs: A 2D tensor, composed of the following (concatenated together).
            - Encoded Z (aka dynamic, aka "controller inputs").
            - (Optional) External Input. Set size with `ext_input_dim`, can be 0.
        states: List of state tensors corresponding to the previous timestep.
            - gen_cell: Generator cell state, of size `units_gen`. Typically initialized from a sample of the f-latent
            distribution (aka static, aka "encoded initial conditions").
            - con_cell: Controller cell state of size `units_con`. Typically initialized with Variable inited to zeros.
            - co x 3: Output only for tracking purposes and external KL loss. Not fed back to next iteration.
                Controller output means, variances, and sampled output (same as means during testing?)
            - factors: The main output. Not fed back to next iteration.
        training: Python boolean indicating whether the layer should behave in
          training mode or in inference mode. Only relevant when `dropout` or
          `recurrent_dropout` is used.
    """
    def __init__(self,
                 units_gen,
                 units_con,
                 factors_dim,
                 co_dim,
                 ext_input_dim,
                 inject_ext_input_to_gen=True,
                 kernel_initializer='lecun_normal',
                 bias_initializer='zeros',
                 recurrent_regularizer='l2',
                 dropout=0.05,
                 clip_value=np.inf,
                 **kwargs):

        self.units_gen = units_gen
        self.units_con = units_con
        self.factors_dim = factors_dim
        self.co_dim = co_dim
        self.ext_input_dim = ext_input_dim
        self.inject_ext_input_to_gen = inject_ext_input_to_gen
        self.units = units_con + units_gen + 3*co_dim + factors_dim
        super().__init__(**kwargs)

        self.dropout = tfkl.Dropout(dropout)
        self.fac_lin = tfkl.Dense(self.factors_dim, use_bias=False,
                                  kernel_initializer='lecun_normal',  # stdev = 1 / np.sqrt(in_size)
                                  kernel_constraint='unit_norm')  # w / sqrt(sum(w**2))
        # Note, we use norm constraint whereas LFADS uses norm on init only.
        if self.units_con > 0:
            self.con_cell = GRUClipCell(self.units_con,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        recurrent_regularizer=recurrent_regularizer,
                                        dropout=dropout,
                                        clip_value=clip_value,
                                        **kwargs)
        else:
            self.con_cell = None

        self.mean_lin = tfkl.Dense(self.co_dim, kernel_initializer='lecun_normal', bias_initializer='zeros')
        self.logvar_lin = tfkl.Dense(self.co_dim, kernel_initializer='lecun_normal', bias_initializer='zeros')

        self.gen_cell = GRUClipCell(self.units_gen,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    dropout=dropout,
                                    clip_value=clip_value,
                                    **kwargs)

    @property
    def state_size(self):
        state_sizes = [self.gen_cell.state_size]
        if self.units_con > 0:
            state_sizes.append(self.con_cell.state_size)
        return tuple(state_sizes) + (self.co_dim,)*3 + (self.factors_dim,)

    @property
    def output_size(self):
        return self.units_con + self.units_gen + 3 * self.co_dim + self.factors_dim

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gen_cell.build(self.co_dim + self.ext_input_dim)
        if self.units_con > 0:
            self.con_cell.build(input_dim + self.factors_dim + self.ext_input_dim)
        self.built = self.gen_cell.built and (self.units_con == 0 or self.con_cell.built)

    def get_config(self):
        config = {
            'units_gen': self.units_gen,
            'units_con': self.units_con,
            'factors_dim': self.factors_dim,
            'co_dim': self.co_dim,
            'ext_input_dim': self.ext_input_dim,
            'inject_ext_input_to_gen': self.inject_ext_input_to_gen
        }
        base_config = super().get_config()
        gru_config = self.gen_cell.get_config()
        return dict(list(base_config.items()) + list(gru_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None, make_K_tensors=True):

        init_state = [self.gen_cell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)]
        if self.units_con > 0:
            init_state += [self.con_cell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)]

        from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
        init_state += [_generate_zero_filled_state(batch_size, self.co_dim, dtype) for _ in range(3)]
        init_state += [_generate_zero_filled_state(batch_size, self.factors_dim, dtype)]
        if make_K_tensors:
            # import tensorflow.keras.backend as K
            # K.is_tensor(init_state[0])
            init_state = [tfkl.Lambda(lambda x: x)(_) for _ in init_state]
        return tuple(init_state)

    def call(self, inputs, states, training=None):
        # if external inputs are used split the inputs
        if self.ext_input_dim > 0:
            con_i = inputs[:, :-self.ext_input_dim]
            ext_inputs = inputs[:, -self.ext_input_dim:]
        else:
            con_i = inputs

        gen_state, con_state = states[:2]

        if self.co_dim > 0:
            # if controller is used
            # input to the controller is (con_i and previous step's factors)
            prev_fac = self.dropout(gen_state, training=training)
            prev_fac = self.fac_lin(prev_fac)
            con_inputs = tf.concat([con_i, prev_fac], axis=1)
            con_inputs = self.dropout(con_inputs, training=training)

            # controller GRU recursion, get new state
            con_outputs, con_s_new = self.con_cell(con_inputs, con_state, training=training)

            # calculate the inputs to the generator
            # transformation to mean and logvar of the posterior
            co_mean = self.mean_lin(con_s_new)
            co_logvar = self.logvar_lin(con_s_new)
            cos_posterior = DiagonalGaussianFromExisting(co_mean, co_logvar)
            if training:  # TODO: (training or "posterior_sample_and_average"), whatever the latter is.
                co_out = cos_posterior.sample
            else:
                co_out = cos_posterior.mean

        else:
            # pass zeros (0-dim) as inputs to generator
            co_out = tf.zeros([tf.shape(input=gen_state)[0], 0])
            con_s_new = co_mean = co_logvar = tf.zeros([tf.shape(input=gen_state)[0], 0])

        # generator's inputs
        if self.ext_input_dim > 0 and self.inject_ext_input_to_gen:
            # passing external inputs along with controller output as generator's input
            gen_inputs = tf.concat([co_out, ext_inputs], axis=1)
        elif self.ext_input_dim > 0 and not self.inject_ext_input_to_gen:
            assert 0, "Not Implemented!"
        else:
            # using only controller output as generator's input
            gen_inputs = co_out

        # generator GRU recursion, get the new state
        gen_outputs, gen_s_new = self.gen_cell(gen_inputs, gen_state, training=training)

        # calculate the factors
        gen_s_new_dropped = self.dropout(gen_s_new, training=training)
        fac_s_new = self.fac_lin(gen_s_new_dropped)

        # Output the states and other values to make them available after RNN
        new_state = [gen_s_new, con_s_new, co_mean, co_logvar, co_out, fac_s_new]
        return new_state, new_state


def test_ComplexCell():
    K.clear_session()

    gen_hidden_state_dim = 100
    con_hidden_state_dim = 30
    factors_dim = 10
    co_dim = 12
    ext_input_dim = 0

    z_enc_dim = 64
    timesteps = 300
    batch_size = 16
    gen_l2_reg = 0.01

    custom_cell = ComplexCell(
        gen_hidden_state_dim,
        con_hidden_state_dim,
        factors_dim,
        co_dim,
        ext_input_dim,
    )

    # Three external pieces of data coming into the generator.
    f_enc = tf.keras.Input(shape=(gen_hidden_state_dim,), name="f_enc")
    z_enc = tf.keras.Input(shape=(timesteps, z_enc_dim), name="z_enc")
    ext_input = tf.keras.Input(shape=(timesteps, ext_input_dim), name="ext_input")

    # Create the RNN.
    generator = tfkl.RNN(custom_cell, return_sequences=True,
                         # recurrent_regularizer=tf.keras.regularizers.l2(l=gen_l2_reg),
                         name='gen_rnn')
    gen_input = tfkl.Concatenate()([z_enc, ext_input])
    gen_init_states = generator.get_initial_state(gen_input)
    # Replace init_states[0] with encoded static state
    gen_init_states[0] = f_enc

    # Run placeholders through RNN
    gen_output = generator(gen_input, initial_state=gen_init_states)
    # Build the model.
    generator_model = tf.keras.Model(inputs=[f_enc, z_enc, ext_input], outputs=gen_output)
    print(generator_model.summary())

    dummy_f_enc = tf.random.uniform((batch_size, gen_hidden_state_dim))
    dummy_z_enc = tf.random.uniform((batch_size, timesteps, z_enc_dim))
    dummy_ext_input = tf.random.uniform((batch_size, timesteps, ext_input_dim))
    gen_s, con_s, co_mean, co_logvar, co_out, fac_s = generator_model([dummy_f_enc, dummy_z_enc, dummy_ext_input])
    print(gen_s.shape, fac_s.shape)


def main():
    test_ComplexCell()


if __name__ == "__main__":
    main()

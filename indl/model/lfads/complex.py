import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from indl.model.lfads.dists import DiagonalGaussianFromExisting


class ComplexCell(tfkl.AbstractRNNCell):
    _BIAS_VARIABLE_NAME = "bias"
    _WEIGHTS_VARIABLE_NAME = "kernel"
    """Cell class for the LFADS Generative GRU + Controller Input

    This cell uses two GRUClipCells: One for the Generator and one for the Controller.
    The Controller  - This is equivalent to the "z2" RNN layer in the other disentangling AE formulations.  
                    - Optional -- only used if z2_units (LFADS: con_dim) > 0
                    - inputs: the concatenation of (a) the encoded controller inputs and (b) the generator cell's state
                      from the previous iteration transformed through the factor dense layer.
                      (on the zeroth step, b starts with f-encoded latents)
                      The encoded controller inputs are themselves the output of an RNN with dim size z1_units,
                      or 'ci_enc_dim' in LFADS
                    - initial state: in LFADS -- a **learnable Variable** of zeros.
    The Generator   - inputs: the output of the Controller cell and optionally 'external' inputs.
                    - initial state: in LFADS -- a sample of a posterior distribution that is parameterized
                      by an encoder.

    The two cells share the same initialization parameters
    (activations, initializers, bias, dropout, regularizer, etc.) except for the number of units.

    Arguments:
        units_gen: Positive integer, number of units in generator RNN cell.
        z2_units: Positive integer, number of units in controller RNN cell. (units_con in LFADS)
        factors_dim: Number of units in Dense layer for factors output.
            This layer would normally be external to the RNN. However, in LFADS, the factors dense layer
            is also used to transform the t-1 generator cell state which becomes part of the _inputs_
            to the controller cell.
        z_latent_size: Dimensionality of variational posterior from controller output --> inputs to controller RNN (LFADS: co_dim)
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
            - Encoded Z1 (LFADS: "controller inputs", other frameworks: half way through dynamic or z-encoding).
            - (Optional) External Input. Set size with `ext_input_dim`, can be 0.
        states: List of state tensors corresponding to the previous timestep.
            - gen_cell: Generator cell state, of size `units_gen`. Typically initialized from a sample of the f-latent
            distribution q(f) (LFADS: "encoded initial conditions"; others: "static").
            - z2_cell: Z2 cell state of size `z2_units`. Initialized with Variable inited to zeros. (LFADS: controller input)
            - z_latent x 3: Output only for tracking purposes and external KL loss. Not fed back to next iteration.
                Controller output means, variances, and sampled output (same as means during *testing*)
            - factors: The main output. Not fed back to next iteration.
        training: Python boolean indicating whether the layer should behave in
          training mode or in inference mode. Only relevant when `dropout` or
          `recurrent_dropout` is used.
    """
    def __init__(self,
                 units_gen,
                 z2_units,
                 factors_dim,
                 z_latent_size,
                 ext_input_dim,
                 inject_ext_input_to_gen=True,
                 kernel_initializer='lecun_normal',
                 bias_initializer='zeros',
                 recurrent_regularizer='l2',
                 dropout=0.05,
                 clip_value=np.inf,
                 **kwargs):

        self.units_gen = units_gen
        self.z2_units = z2_units
        self.factors_dim = factors_dim
        self.z_latent_size = z_latent_size
        self.ext_input_dim = ext_input_dim
        self.inject_ext_input_to_gen = inject_ext_input_to_gen
        self.units = z2_units + units_gen + 3*z_latent_size + factors_dim
        super().__init__(**kwargs)

        self.dropout = tfkl.Dropout(dropout)
        self.fac_lin = tfkl.Dense(self.factors_dim, use_bias=False,
                                  kernel_initializer='lecun_normal',  # stdev = 1 / np.sqrt(in_size)
                                  kernel_constraint='unit_norm')  # w / sqrt(sum(w**2))
        # Note, we use norm constraint whereas LFADS uses norm on init only.
        from indl.rnn.gru_clip import GRUClipCell

        if self.z2_units > 0:
            self.z2_cell = GRUClipCell(self.z2_units,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       recurrent_regularizer=recurrent_regularizer,
                                       dropout=dropout,
                                       clip_value=clip_value,
                                       **kwargs)
        else:
            self.z2_cell = None

        self.mean_lin = tfkl.Dense(self.z_latent_size, kernel_initializer='lecun_normal', bias_initializer='zeros')
        self.logvar_lin = tfkl.Dense(self.z_latent_size, kernel_initializer='lecun_normal', bias_initializer='zeros')

        self.gen_cell = GRUClipCell(self.units_gen,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    dropout=dropout,
                                    clip_value=clip_value,
                                    **kwargs)

    @property
    def state_size(self):
        # [gen_s_new, z2_state, z_latent_mean, z_latent_logvar, q_z_sample, factors_new]
        state_sizes = [self.gen_cell.state_size]
        if self.z2_units > 0:
            state_sizes.append(self.z2_cell.state_size)
        return tuple(state_sizes) + (self.z_latent_size,)*3 + (self.factors_dim,)

    @property
    def output_size(self):
        return self.z2_units + self.units_gen + 3 * self.z_latent_size + self.factors_dim

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.z2_units > 0:
            self.z2_cell.build(input_dim + self.factors_dim + self.ext_input_dim)
        self.gen_cell.build(self.z_latent_size + self.ext_input_dim)
        self.built = (self.z2_units == 0 or self.z2_cell.built) and self.gen_cell.built

    def get_config(self):
        config = {
            'units_gen': self.units_gen,
            'z2_units': self.z2_units,
            'factors_dim': self.factors_dim,
            'z_latent_size': self.z_latent_size,
            'ext_input_dim': self.ext_input_dim,
            'inject_ext_input_to_gen': self.inject_ext_input_to_gen
        }
        base_config = super().get_config()
        gru_config = self.gen_cell.get_config()
        return dict(list(base_config.items()) + list(gru_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None, make_K_tensors=True):

        init_state = [self.gen_cell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)]
        if self.z2_units > 0:
            init_state += [self.z2_cell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)]

        from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
        init_state += [_generate_zero_filled_state(batch_size, self.z_latent_size, dtype) for _ in range(3)]
        init_state += [_generate_zero_filled_state(batch_size, self.factors_dim, dtype)]
        if make_K_tensors:
            # import tensorflow.keras.backend as K
            # K.is_tensor(init_state[0])
            init_state = [tfkl.Lambda(lambda x: x)(_) for _ in init_state]
        return tuple(init_state)

    def call(self, inputs, states, training=None):
        if training is None:
            training = K.learning_phase()
        # if external inputs are used split the inputs
        if self.ext_input_dim > 0:
            z1 = inputs[:, :-self.ext_input_dim]
            ext_inputs = inputs[:, -self.ext_input_dim:]
        else:
            z1 = inputs
            ext_inputs = None

        gen_state, z2_state = states[:2]

        if self.z_latent_size > 0:
            # if controller is used
            # input to the controller is (con_i and previous step's factors)
            prev_gen_dropped = self.dropout(gen_state, training=training)
            prev_fac = self.fac_lin(prev_gen_dropped)
            z2_inputs = tf.concat([z1, prev_fac], axis=1)
            z2_inputs = self.dropout(z2_inputs, training=training)

            # controller GRU recursion, get new state
            z2_outputs, z2_state = self.z2_cell(z2_inputs, z2_state, training=training)

            # calculate the inputs to the generator
            # transformation to mean and logvar of the posterior
            # TODO: use make_variational(params, z2_state)
            z_latent_mean = self.mean_lin(z2_state)
            z_latent_logvar = self.logvar_lin(z2_state)
            z_latent_dist = DiagonalGaussianFromExisting(z_latent_mean, z_latent_logvar)
            if training:  # TODO: (training or "posterior_sample_and_average"), whatever the latter is.
                q_z_sample = z_latent_dist.sample
            else:
                q_z_sample = z_latent_dist.mean

        else:
            # pass zeros (0-dim) as inputs to generator
            q_z_sample = tf.zeros([tf.shape(input=gen_state)[0], 0])
            z2_state = z_latent_mean = z_latent_logvar = tf.zeros([tf.shape(input=gen_state)[0], 0])

        # generator's inputs
        if self.ext_input_dim > 0 and self.inject_ext_input_to_gen:
            # passing external inputs along with controller output as generator's input
            gen_inputs = tf.concat([q_z_sample, ext_inputs], axis=1)
        elif self.ext_input_dim > 0 and not self.inject_ext_input_to_gen:
            assert 0, "Not Implemented!"
        else:
            # using only controller output as generator's input
            gen_inputs = q_z_sample

        # generator GRU recursion, get the new state
        gen_outputs, gen_s_new = self.gen_cell(gen_inputs, gen_state, training=training)

        # calculate the factors
        gen_s_new_dropped = self.dropout(gen_s_new, training=training)
        factors_new = self.fac_lin(gen_s_new_dropped)

        # Output the states and other values to make them available after RNN
        new_state = [gen_s_new, z2_state, z_latent_mean, z_latent_logvar, q_z_sample, factors_new]
        return new_state, new_state

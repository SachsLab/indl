import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.util import nest
from tensorflow.python.keras.layers.recurrent import _standardize_args
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
from indl.model.generative import GenerativeRNN
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
import numpy as np
scale_shift = np.log(np.exp(1) - 1).astype(np.float32)


class MVNDiagLSTMCell(tfkl.LSTMCell):

    def __init__(self, units,
                 dist_ev_size=None,
                 **kwargs):
        super(MVNDiagLSTMCell, self).__init__(units, **kwargs)
        self.dist_ev_size = dist_ev_size or units
        self.loc = tfkl.Dense(self.dist_ev_size)
        self.unxf_scale = tfkl.Dense(self.dist_ev_size)
        self.dist_layer = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1]),
            # TODO: convert_to_tensor_fn=lambda s: s.sample(N_SAMPLES)
        )

    def call(self, inputs, states, training=None):
        inputs = tf.convert_to_tensor(inputs)
        output, state = super(MVNDiagLSTMCell, self).call(inputs, states, training=training)
        loc = self.loc(output)
        unxf_scale = self.unxf_scale(output)
        scale = tf.nn.softplus(unxf_scale + scale_shift) + 1e-5
        dist = self.dist_layer([loc, scale])
        return dist, state


if __name__ == "__main__":
    N_UNITS = 24
    N_OUT_TIMESTEPS = 115
    cell = MVNDiagLSTMCell  # LSTMCell or GRU

    # Test placeholder tensor with no timesteps
    K.clear_session()
    gen_rnn_layer = GenerativeRNN(cell(N_UNITS), return_sequences=True, return_state=True,
                                  timesteps=N_OUT_TIMESTEPS)
    in_ = tfkl.Input(shape=(N_UNITS,))
    x_, cell_state_ = gen_rnn_layer(in_)
    print("Test placeholder tensor")
    
    model = tf.keras.Model(inputs=in_, outputs=x_)
    model.summary()

    # Test None input --> uses zeros
    K.clear_session()
    gen_rnn_layer = GenerativeRNN(cell(N_UNITS), return_sequences=True, return_state=True,
                                  timesteps=N_OUT_TIMESTEPS)
    print("Test None input")
    x_, cell_state_ = gen_rnn_layer()

    # Test random input
    K.clear_session()
    gen_rnn_layer = GenerativeRNN(cell(N_UNITS), return_sequences=True, return_state=True,
                                  timesteps=N_OUT_TIMESTEPS)
    in_ = tf.random.uniform((1, 8, N_UNITS))
    print("Test zeros input")
    x_, cell_state_ = gen_rnn_layer(in_)

    # Test random states
    K.clear_session()
    gen_rnn_layer = GenerativeRNN(cell(N_UNITS), return_sequences=True, return_state=True,
                                  timesteps=N_OUT_TIMESTEPS)
    print(gen_rnn_layer.compute_output_shape())
    init_states = [tf.random.uniform((1, N_UNITS), minval=-1.0, maxval=1.0) for _ in range(1)]
    x_, cell_states_ = gen_rnn_layer(initial_state=init_states)
    print(x_, cell_states_)

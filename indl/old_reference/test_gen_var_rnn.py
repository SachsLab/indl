import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors


class VariationalLSTMCell(tfkl.LSTMCell):
    def __init__(self, units,
                 make_dist_fn=None,
                 make_dist_model=None,
                 output_dim=None, **kwargs):
        super(VariationalLSTMCell, self).__init__(units, **kwargs)
        self.make_dist_fn = make_dist_fn
        self.output_dim = output_dim or units
        self.make_dist_model = make_dist_model

        # For some reason the below code doesn't work during build.
        # So I don't know how to use the outer VariationalRNN to set this cell's output_dim
        if self.make_dist_fn is None:
            self.make_dist_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1])
        if self.make_dist_model is None:
            fake_cell_output = tfkl.Input((self.units,))
            loc = tfkl.Dense(self.output_dim)(fake_cell_output)
            scale = tfkl.Dense(self.output_dim)(fake_cell_output)
            scale = tf.nn.softplus(scale + np.log(np.exp(1) - 1)) + 1e-5
            dist_layer = tfpl.DistributionLambda(
                make_distribution_fn=self.make_dist_fn,
                # TODO: convert_to_tensor_fn=lambda s: s.sample(N_SAMPLES)
            )([loc, scale])
            self.make_dist_model = tf.keras.Model(fake_cell_output, dist_layer)

    def build(self, input_shape):
        super(VariationalLSTMCell, self).build(input_shape)
        # It would be good to defer making self.make_dist_model until here, but it doesn't work for some reason.

    def input_zero(self, inputs_):
        input0 = inputs_[..., -1, :]
        input0 = tf.matmul(input0, tf.zeros((input0.shape[-1], self.units)))
        dist0 = self.make_dist_model(input0)
        return dist0

    def call(self, inputs, states, training=None):
        inputs = tf.convert_to_tensor(inputs)
        output, state = super(VariationalLSTMCell, self).call(inputs, states, training=training)
        dist = self.make_dist_model(output)
        return dist, state


class RNNMultivariateNormalDiag(tfd.MultivariateNormalDiag):
    def __init__(self, cell, n_timesteps=1, output_dim=None, name="rnn_mvn_diag", **kwargs):
        self.cell = cell
        if output_dim is not None and hasattr(self.cell, 'output_dim'):
            self.cell.output_dim = output_dim
        if hasattr(self.cell, 'output_dim'):
            output_dim = self.cell.output_dim
        else:
            output_dim = output_dim or self.cell.units

        h0 = tf.zeros([1, self.cell.units])
        c0 = tf.zeros([1, self.cell.units])
        input0 = tf.zeros((1, output_dim))

        if hasattr(cell, 'reset_dropout_mask'):
            self.cell.reset_dropout_mask()
            self.cell.reset_recurrent_dropout_mask()

        input_ = input0
        states_ = (h0, c0)
        successive_outputs = []
        for i in range(n_timesteps):
            input_, states_ = self.cell(input_, states_)
            successive_outputs.append(input_)

        loc = tf.concat([_.parameters["distribution"].parameters["loc"]
                         for _ in successive_outputs],
                        axis=0)
        scale_diag = tf.concat([_.parameters["distribution"].parameters["scale_diag"]
                                for _ in successive_outputs],
                               axis=0)

        super(RNNMultivariateNormalDiag, self).__init__(loc=loc, scale_diag=scale_diag, name=name, **kwargs)


class GenerativeVariationalRNN(tfkl.RNN):
    """
    Generative because each call starts with a zero-state,
    and feeds the previous iteration to the next iteration.

    Variational because it returns a distribution, parameterized
    by the output of the cell.
    """

    def __init__(self, cell, output_dim=None, **kwargs):
        # TODO: kwargs for creation of output distribution
        #  --> should this always be shared with cell?
        super(GenerativeVariationalRNN, self).__init__(cell, **kwargs)
        self.supports_masking = False
        self._supports_ragged_inputs = False

        if output_dim is not None and hasattr(self.cell, 'output_dim'):
            self.cell.output_dim = output_dim
        if hasattr(self.cell, 'output_dim'):
            self.output_dim = self.cell.output_dim
        else:
            self.output_dim = output_dim or self.cell.units

    def compute_output_shape(self, input_shape):
        print(f"compute_output_shape - input_shape: {input_shape}")
        super(GenerativeVariationalRNN, self).compute_output_shape(input_shape)

    def build(self, input_shape):
        print(f"build - input_shape: {input_shape}")
        super(GenerativeVariationalRNN, self).build(input_shape)

    def call(self, inputs, training=None, initial_state=None, constants=None):
        input_shape = K.int_shape(inputs)
        gen_timesteps = input_shape[0] if self.time_major else input_shape[1]

        _, initial_state, constants = self._process_inputs(
            inputs, initial_state, constants)

        if hasattr(self.cell, 'input_zero'):
            input0 = self.cell.input_zero(inputs)
        else:
            input0 = inputs[..., -1, :]
            input0 = tf.matmul(input0, tf.zeros((input0.shape[-1], self.units)))
        states = tuple(initial_state)

        self._maybe_reset_cell_dropout_mask(self.cell)

        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        # TF RNN cells expect single tensor as state instead of list wrapped tensor.
        is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None

        def step(inputs_, states_):
            states_ = states_[0] if len(states_) == 1 and is_tf_rnn_cell else states_
            output, new_states = self.cell.call(inputs_, states_, **kwargs)
            if not nest.is_sequence(new_states):
                new_states = [new_states]
            return output, new_states

        batch = input0.shape[0]
        input0.shape.with_rank_at_least(2)
        input = input0
        successive_states = []
        successive_outputs = []
        for i in range(gen_timesteps):
            input, states = step(input, states)
            successive_outputs.append(input)
            successive_states.append(states)
        last_output = successive_outputs[-1]
        states = successive_states[-1]

        # If output is distribution then its parameterizations can be used directly
        if issubclass(type(successive_outputs[0]), tfp.distributions.Distribution):
            # TODO: Generalize for dists other than MVNDiag
            if not self.return_sequences:
                loc = last_output.parameters["distribution"].parameters["loc"]
                scale_diag = last_output.parameters["distribution"].parameters["scale_diag"]
            else:
                stack_ax = 0 if self.time_major else 1
                loc = tf.stack([_.parameters["distribution"].parameters["loc"] for _ in successive_outputs], axis=stack_ax)
                scale_diag = tf.stack([_.parameters["distribution"].parameters["scale_diag"] for _ in successive_outputs], axis=stack_ax)
            output = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        else:
            outputs = array_ops.stack(successive_outputs)
            # static shape inference
            def set_shape(output_):
                if isinstance(output_, tf.python.framework.ops.Tensor):
                    shape = output_.shape.as_list()
                    shape[0] = gen_timesteps
                    shape[1] = batch
                    output_.set_shape(shape)
                return output_
            outputs = nest.map_structure(set_shape, outputs)
            if not self.return_sequences:
                outputs = last_output
            else:
                if not self.time_major:
                    def swap_batch_timestep(input_t):
                        # Swap the batch and timestep dim for the incoming tensor.
                        axes = list(range(len(input_t.shape)))
                        axes[0], axes[1] = 1, 0
                        return array_ops.transpose(input_t, axes)
                    outputs = nest.map_structure(swap_batch_timestep, outputs)
            loc = tfkl.Dense(self.output_dim)(outputs)
            scale = tfkl.Dense(self.output_dim)(outputs)
            scale = tf.nn.softplus(scale + np.log(np.exp(1) - 1)) + 1e-5
            output = tfpl.DistributionLambda(make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                loc=t[0], scale_diag=t[1]))([loc, scale])

        if self.stateful:
            updates = []
            for state_, state in zip(nest.flatten(self.states), nest.flatten(states)):
                updates.append(state_ops.assign(state_, state))
            self.add_update(updates)

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return generic_utils.to_list(output) + states
        else:
            return output


if __name__ == "__main__":

    N_HIDDEN = 6
    N_LATENT = 4
    N_SAMPLES = 3
    n_timesteps = 50
    BATCH_SIZE = 8

    # test_input = tfkl.Input((n_timesteps, N_LATENT))
    # test_input = tf.zeros((BATCH_SIZE, n_timesteps, N_LATENT))

    # test_layer = GenerativeVariationalRNN(
    #     VariationalLSTMCell(N_HIDDEN, output_dim=N_LATENT), return_sequences=True,
    # )
    # test_layer = tfkl.RNN(
    #     VariationalLSTMCell(N_HIDDEN, output_dim=N_LATENT), return_sequences=True
    # )
    # out_dist = test_layer(test_input)

    rnn_mvn_prior = RNNMultivariateNormalDiag(VariationalLSTMCell(N_HIDDEN, output_dim=N_LATENT),
                                              n_timesteps=n_timesteps, output_dim=N_LATENT)
    print(rnn_mvn_prior)
    samp = rnn_mvn_prior.sample()
    print(samp)

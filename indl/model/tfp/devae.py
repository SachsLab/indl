import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from indl.model.tfp import LearnableMultivariateNormalDiagCell, LearnableMultivariateNormalDiag
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
scale_shift = np.log(np.exp(1) - 1).astype(np.float32)


# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/disentangled_vae.py


class VariationalLSTMCell(tfkl.LSTMCell):

    def __init__(self, units,
                 make_dist_fn=None,
                 make_dist_model=None,
                 **kwargs):
        super(VariationalLSTMCell, self).__init__(units, **kwargs)
        self.make_dist_fn = make_dist_fn
        self.make_dist_model = make_dist_model

        # For some reason the below code doesn't work during build.
        # So I don't know how to use the outer VariationalRNN to set this cell's output_size
        if self.make_dist_fn is None:
            self.make_dist_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1])
        if self.make_dist_model is None:
            fake_cell_output = tfkl.Input((self.units,))
            loc = tfkl.Dense(self.output_size, name="VarLSTMCell_loc")(fake_cell_output)
            scale = tfkl.Dense(self.output_size, name="VarLSTMCell_scale")(fake_cell_output)
            scale = tf.nn.softplus(scale + scale_shift) + 1e-5
            dist_layer = tfpl.DistributionLambda(
                make_distribution_fn=self.make_dist_fn,
                # TODO: convert_to_tensor_fn=lambda s: s.sample(N_SAMPLES)
            )([loc, scale])
            self.make_dist_model = tf.keras.Model(fake_cell_output, dist_layer)

    def build(self, input_shape):
        super(VariationalLSTMCell, self).build(input_shape)
        # It would be good to defer making self.make_dist_model until here,
        # but it doesn't work for some reason.

    # def input_zero(self, inputs_):
    #    input0 = inputs_[..., -1, :]
    #    input0 = tf.matmul(input0, tf.zeros((input0.shape[-1], self.units)))
    #    dist0 = self.make_dist_model(input0)
    #    return dist0

    def call(self, inputs, states, training=None):
        inputs = tf.convert_to_tensor(inputs)
        output, state = super(VariationalLSTMCell, self).call(inputs, states, training=training)
        dist = self.make_dist_model(output)
        return dist, state


class DynamicEncoder(tf.keras.Model):
    def __init__(self, units, n_times, output_dim, name="dynamic_encoder"):
        super(DynamicEncoder, self).__init__(name=name)
        self.dynamic_prior_cell = LearnableMultivariateNormalDiagCell(units, output_dim)
        self.n_times = n_times
        self.loc = tfkl.Dense(output_dim, name="loc")
        self.unxf_scale = tfkl.Dense(output_dim, name="scale")
        self.q_z_layer = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1]),
            name="q_z"
        )

    def call(self, inputs):
        # Assume inputs doesn't have time-axis. Broadcast-add zeros to add time axis.
        inputs_ = inputs[..., tf.newaxis, :] + tf.zeros([self.n_times, 1])
        loc = self.loc(inputs_)
        unxf_scale = self.unxf_scale(inputs_)
        scale = tf.math.softplus(unxf_scale + scale_shift) + 1e-5
        q_z = self.q_z_layer([loc, scale])
        # _, dynamic_prior = self.sample_dynamic_prior(self.n_times)
        # kld = tfd.kl_divergence(q_z, dynamic_prior)
        # kld = tf.reduce_sum(kld, axis=-1)
        # kld = tf.reduce_mean(kld)
        # self.add_loss(KL_WEIGHT * kld)
        return q_z

    def sample_dynamic_prior(self, steps, samples=1, batches=1, fixed=False):
        """Samples LSTM cell->MVNDiag for each steps

        Args:
          steps: Number of timesteps to sample for each sequence.
          samples: Number of samples to draw from the latent distribution.
          batch_size: Number of sequences to sample.
          fixed: Boolean for whether or not to share the same random
            sample across all sequences.

        Returns:
          A tuple of a sample tensor of shape [samples, batch_size, steps,
          latent_size], and a MultivariateNormalDiag distribution from which
          the tensor was sampled, with event shape [latent_size], and batch
          shape [samples, 1, length] if fixed or [samples, batch_size,
          length] otherwise.
        """
        if fixed:
            sample_batch_size = 1
        else:
            sample_batch_size = batches

        sample, state = self.dynamic_prior_cell.zero_state([samples, sample_batch_size])
        locs = []
        scale_diags = []
        sample_list = []
        for _ in range(steps):
            dist, state = self.dynamic_prior_cell(sample, state)
            sample = dist.sample()
            locs.append(dist.parameters["loc"])
            scale_diags.append(dist.parameters["scale_diag"])
            sample_list.append(sample)

        sample = tf.stack(sample_list, axis=2)
        loc = tf.stack(locs, axis=2)
        scale_diag = tf.stack(scale_diags, axis=2)

        if fixed:  # tile along the batch axis
            sample = sample + tf.zeros([batches, 1, 1])

        return sample, tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class StaticEncoder(tf.keras.Model):
    def __init__(self, latent_size, name="static_encoder"):
        super(StaticEncoder, self).__init__(name=name)
        self.static_prior_factory = LearnableMultivariateNormalDiag(latent_size)
        self.loc = tfkl.Dense(latent_size, name="loc")
        self.unxf_scale = tfkl.Dense(
            tfpl.MultivariateNormalTriL.params_size(latent_size) - latent_size,
            name="scale")
        self.scale_bijector = tfp.bijectors.FillScaleTriL()
        self.q_f_layer = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalTriL(loc=t[0], scale_tril=t[1]),
            name="q_f"
        )

    def call(self, inputs):
        loc = self.loc(inputs)
        unxf_scale = self.unxf_scale(inputs)
        scale = self.scale_bijector(unxf_scale)
        q_f = self.q_f_layer([loc, scale])
        # static_prior = self.static_prior_factory()
        # kld = tfd.kl_divergence(q_f, static_prior)
        # kld = tf.reduce_mean(kld)
        # self.add_loss(KL_WEIGHT * kld)
        return q_f


class Decoder(tf.keras.Model):
    def __init__(self, n_times, out_dim, name='decoder'):
        super(FactorizedDecoder, self).__init__(name=name)
        self.n_times = n_times
        self.concat = tfkl.Concatenate()
        self.loc = tfkl.Dense(out_dim, name="loc")
        self.unxf_scale = tfkl.Dense(out_dim, name="scale")
        self.q_z_layer = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1]),
            name="p_out"
        )

    def call(self, inputs):
        f_sample = inputs[0][..., tf.newaxis, :] + tf.zeros([self.n_times, 1])
        z_sample = tf.convert_to_tensor(inputs[1])
        y = self.concat([z_sample, f_sample])
        loc = self.loc(y)
        unxf_scale = self.unxf_scale(y)
        scale = tf.math.softplus(unxf_scale + scale_shift) + 1e-5
        p_out = self.q_z_layer([loc, scale])
        return p_out


class FactorizedAutoEncoder(tf.keras.Model):
    def __init__(self, units, n_times, latent_size_static, latent_size_dynamic, n_out_dim,
                 name='autoencoder'):
        super(FactorizedAutoEncoder, self).__init__(name=name)
        self.static_encoder = StaticEncoder(latent_size_static)
        self.dynamic_encoder = DynamicEncoder(units, n_times, latent_size_dynamic)
        self.decoder = Decoder(n_times, n_out_dim)

    def call(self, inputs):
        q_f = self.static_encoder(inputs)
        q_z = self.dynamic_encoder(inputs)
        p_out = self.decoder([tf.convert_to_tensor(q_f),
                              tf.convert_to_tensor(q_z)])
        return p_out


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

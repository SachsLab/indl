from typing import Union, List
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
from indl.utils import scale_shift
from indl.dists import make_learnable_mvn_params
from indl.dists import LearnableMultivariateNormalCell


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


class IProcessMVNGenerator:

    def __init__(self):
        pass

    def get_dist(self, timesteps):

        raise NotImplementedError


class AR1ProcessMVNGenerator(IProcessMVNGenerator):
    """
    Similar to LFADS' LearnableAutoRegressive1Prior.
    Here we use the terminology from:
    https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process

    The autoregressive function takes the form:
        E(X_t) = E(c) + phi * E(X_{t-1}) + e_t
    E(c) is a constant.
    phi is a parameter, which is equivalent to exp(-1/tau) = exp(-exp(-logtau)).
        where tau is a time constant.
    e_t is white noise with zero-mean with evar = sigma_e**2

    When there's no previous sample, E(X_t) = E(c) + e_t,
        which is a draw from N(c, sigma_e**2)
    When there is a previous sample, E(X_t) = E(c) + phi * E(X_{t-1}) + e_t,
        which means a draw from N(c + phi * X_{t-1}, sigma_p**2)
        where sigma_p**2 = phi**2 * var(X_{t-1}) + sigma_e**2 = sigma_e**2 / (1 - phi**2)
        or logpvar = logevar - (log(1 - phi) + log(1 + phi))

    Note that this could be roughly equivalent to tfd.Autoregressive if it was passed
    a `distribution_fn` with the same transition.

    See issue: https://github.com/snel-repo/lfads-cd/issues/1
    """
    def __init__(self, init_taus: Union[float, List[float]],
                 init_std: Union[float, List[float]] = 0.1,
                 trainable_mean: bool = False,
                 trainable_tau: bool = True,
                 trainable_var: bool = True,
                 offdiag: bool = False):
        """


        Args:
            init_taus: Initial values of tau
            init_std: Initial value of sigma_e
            trainable_mean: set True if the mean (e_c) is trainable.
            trainable_tau: set True to
            trainable_nvar:
        """
        self._offdiag = offdiag
        if isinstance(init_taus, float):
            init_taus = [init_taus]
        # TODO: Add time axis for easier broadcasting
        ndim = len(init_taus)
        self._e_c, self._e_scale = make_learnable_mvn_params(ndim, init_std=init_std,
                                                             trainable_mean=trainable_mean,
                                                             trainable_var=trainable_var,
                                                             offdiag=offdiag)
        self._logtau = tf.Variable(tf.math.log(init_taus), dtype=tf.float32, trainable=trainable_tau)
        self._phi = tf.exp(-tf.exp(-self._logtau))
        self._p_scale = tf.exp(tf.math.log(self._e_scale) - (tf.math.log(1 - self._phi) + tf.math.log(1 + self._phi)))

    def get_dist(self, timesteps, samples=1, batch_size=1, fixed=False):
        locs = []
        scales = []
        sample_list = []

        # Add a time dimension
        e_c = tf.expand_dims(self._e_c, 0)
        e_scale = tf.expand_dims(self._e_scale, 0)
        p_scale = tf.expand_dims(self._p_scale, 0)

        sample = tf.expand_dims(tf.expand_dims(tf.zeros_like(e_c), 0), 0)
        sample = tf.tile(sample, [samples, batch_size, 1, 1])
        for _ in range(timesteps):
            loc = e_c + self._phi * sample
            scale = p_scale if _ > 0 else e_scale
            locs.append(loc)
            scales.append(scale)
            if self._offdiag:
                dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale)
            else:
                dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            sample = dist.sample()
            sample_list.append(sample)

        sample = tf.concat(sample_list, axis=2)
        loc = tf.concat(locs, axis=2)
        scale = tf.concat(scales, axis=-2)
        if self._offdiag:
            dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale)
        else:
            dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        return sample, dist


class RNNMVNGenerator(IProcessMVNGenerator):
    """
    Similar to DSAE's LearnableMultivariateNormalDiagCell
    """
    def __init__(self, units: int, out_dim: int, cell_type: str, shift_std: float = 0.1, offdiag: bool = False):
        """

        Args:
            units: Dimensionality of the RNN function parameters.
            out_dim: The dimensionality of the distribution.
            cell_type: an RNN cell type among 'lstm', 'gru', 'rnn', 'gruclip'. case-insensitive.
            shift_std: Shift applied to MVN std before building the dist. Providing a shift
                toward the expected std allows the input values to be closer to 0.
            offdiag: set True to allow non-zero covariance (within-timestep) in the returned distribution.
        """
        self.cell = LearnableMultivariateNormalCell(units, out_dim, cell_type=cell_type,
                                                    shift_std=shift_std, offdiag=offdiag)

    def get_dist(self, timesteps, samples=1, batch_size=1, fixed=True):
        """
        Samples from self.cell `timesteps` times.
        On each step, the previous (sample, state) is fed back into the cell
        (zero_state used for 0th step).

        The cell returns a multivariate normal diagonal distribution for each timestep.
        We collect each timestep-dist's params (loc and scale), then use them to create
        the return value: a single MVN diag dist that has a dimension for timesteps.

        The cell returns a full dist for each timestep so that we can 'sample' it.
        If our sample size is 1, and our cell is an RNN cell, then this is roughly equivalent
        to doing a generative RNN (init state = zeros, return_sequences=True) then passing
        those values through a pair of Dense layers to parameterize a single MVNDiag.

        Args:
            timesteps: Number of times to sample from the dynamic_prior_cell. Output will have
            samples: Number of samples to draw from the latent distribution.
            batch_size: Number of sequences to sample.
            fixed: Boolean for whether or not to share the same random
                    sample across all sequences in batch.
https://github.com/tensorflow/probability/blob/698e0101aecf46c42858db7952ee3024e091c291/tensorflow_probability/examples/disentangled_vae.py#L887
        Returns:

        """
        if fixed:
            sample_batch_size = 1
        else:
            sample_batch_size = batch_size

        sample, state = self.cell.zero_state([samples, sample_batch_size])
        locs = []
        scales = []
        sample_list = []
        scale_parm_name = "scale_tril" if self.cell.offdiag else "scale_diag"  # TODO: Check this for offdiag
        for _ in range(timesteps):
            dist, state = self.cell(sample, state)
            sample = dist.sample()
            locs.append(dist.parameters["loc"])
            scales.append(dist.parameters[scale_parm_name])
            sample_list.append(sample)

        sample = tf.stack(sample_list, axis=2)
        loc = tf.stack(locs, axis=2)
        scale = tf.stack(scales, axis=2)

        if fixed:  # tile along the batch axis
            sample = sample + tf.zeros([batch_size, 1, 1])

        if self.cell.offdiag:
            dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale)
        else:
            dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        return sample, dist


class TiledMVNGenerator(IProcessMVNGenerator):
    """
    Similar to LFADS' LearnableDiagonalGaussian.
    Uses a single learnable loc and scale which are tiled across timesteps.
    """
    def __init__(self, latent_dim: int, init_std: float = 0.1,
                 trainable_mean: bool = True, trainable_var: bool = True,
                 offdiag: bool = False):
        """

        Args:
            latent_dim: Number of dimensions in a single timestep  (params['f_latent_size'])
            init_std: Initial value of standard deviation (params['q_z_init_std'])
            trainable_mean: True if mean should be trainable (params['z_prior_train_mean'])
            trainable_var: True if variance should be trainable (params['z_prior_train_var'])
            offdiag: True if off-diagonal elements (non-orthogonality) allowed. (params['z_prior_off_diag'])
        """
        self._offdiag = offdiag
        self._loc, self._scale = make_learnable_mvn_params(latent_dim, init_std=init_std,
                                                           trainable_mean=trainable_mean,
                                                           trainable_var=trainable_var,
                                                           offdiag=offdiag)

    def get_dist(self, timesteps, samples=1, batch_size=1):
        """
        Tiles the saved loc and scale to the same shape as `posterior` then uses them to
        create a MVN dist with appropriate shape. Each timestep has the same loc and
        scale but if it were sampled then each timestep would return different values.
        Args:
            timesteps:
            samples:
            batch_size:
        Returns:
            MVNDiag distribution of the same shape as `posterior`
        """
        loc = tf.tile(tf.expand_dims(self._loc, 0), [timesteps, 1])
        scale = tf.expand_dims(self._scale, 0)
        if self._offdiag:
            scale = tf.tile(scale, [timesteps, 1, 1])
            dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale)
        else:
            scale = tf.tile(scale, [timesteps, 1])
            dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        return dist.sample([samples, batch_size]), dist

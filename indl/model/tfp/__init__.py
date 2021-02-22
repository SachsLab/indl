from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
scale_shift = np.log(np.exp(1) - 1).astype(np.float32)


def make_mvn_prior(ndim: int, init_std: float = 1.0, trainable_mean: bool = True, trainable_var: bool = True,
                   offdiag: bool = False) -> Union[tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL]:
    """
    Creates a tensorflow probability distribution:
        MultivariateNormalTriL if offdiag else MultivariateNormalDiag
    Mean (loc) and sigma (scale) can be trainable or not.
    Mean initializes to random.normal around 0 (stddev=0.1) if trainable, else zeros.
    Scale initialies to init_std if not trainable. If it is trainable, it initializes
     to a tfp TransformedVariable that will be centered at 0 for easy training under the hood,
     but will be transformed via softplus to give something initially close to init_var.

    loc and scale are tracked by the MVNDiag class.

    For LFADS ics prior, trainable_mean=True, trainable_var=False
    For LFADS cos prior (if not using AR1), trainable_mean=False, trainable_var=True
    In either case, var was initialized with 0.1 (==> logvar with log(0.1))
    Unlike the LFADS' LearnableDiagonalGaussian, here we don't support multi-dimensional, just a vector.

    See also LearnableMultivariateNormalDiag for a tf.keras.Model version of this.

    Args:
        ndim: latent dimension of distribution. Currently only supports 1 d (I think )
        init_std: initial standard deviation of the gaussian. If trainable_var then the initial standard deviation
            will be drawn from a random.normal distribution with mean init_std and stddev 1/10th of that.
        trainable_mean: If the mean should be a tf.Variable
        trainable_var: If the variance/stddev/scale (whatever you call it) should be a tf.Variable
        offdiag: If the variance-covariance matrix is allowed non-zero off-diagonal elements.

    Returns:
        A tensorflow-probability distribution (either MultivariateNormalTriL or MultivariateNormalDiag).
    """
    if trainable_mean:
        loc = tf.Variable(tf.random.normal([ndim], stddev=0.1, dtype=tf.float32))
    else:
        loc = tf.zeros(ndim)
    # Initialize the variance (scale), trainable or not, offdiag or not.
    if trainable_var:
        if offdiag:
            _ndim = [ndim, ndim]
            scale = tfp.util.TransformedVariable(
                # init_std * tf.eye(ndim, dtype=tf.float32),
                tf.random.normal(_ndim, mean=init_std, stddev=init_std/10, dtype=tf.float32),
                tfp.bijectors.FillScaleTriL(),
                name="prior_scale")
        else:
            _scale_shift = np.log(np.exp(init_std) - 1).astype(np.float32)  # tfp.math.softplus_inverse(init_std)
            scale = tfp.util.TransformedVariable(
                # init_std * tf.ones(ndim, dtype=tf.float32),
                tf.random.normal([ndim], mean=init_std, stddev=init_std/10, dtype=tf.float32),
                tfb.Chain([tfb.Shift(1e-5), tfb.Softplus(), tfb.Shift(_scale_shift)]),
                name="prior_scale")
    else:
        if offdiag:
            scale = init_std * tf.eye(ndim)
        else:
            scale = init_std * tf.ones(ndim)

    # Initialize the prior.
    if offdiag:
        # Note: Diag must be > 0, upper triangular must be 0, and lower triangular may be != 0.
        prior = tfd.MultivariateNormalTriL(
            loc=loc,
            scale_tril=scale
        )
    else:
        prior = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        # kl_exact needs same dist types for prior and latent.
        # We would switch to the next line if we switched our latent to using tf.Independent(tfd.Normal)
        # prior = tfd.Independent(tfd.Normal(loc=tf.zeros(ndim), scale=1), reinterpreted_batch_ndims=1)

    return prior


def make_mvn_dist_fn(_x_, ndim, shift_std=1.0, offdiag=False, loc_name=None, scale_name=None, use_mvn_diag=True):
    """
    Take a 1-D tensor and use it to parameterize a MVN dist.
    This doesn't return the distribution, but the function to make the distribution and its arguments.
    make_dist_fn, [loc, scale]
    You can supply it to tfpl.DistributionLambda

    Args:
        _x_:
        ndim:
        shift_std:
        offdiag:
        loc_name:
        scale_name:
        use_mvn_diag:

    Returns:

    """

    _scale_shift = np.log(np.exp(shift_std) - 1).astype(np.float32)
    _loc = tfkl.Dense(ndim, name=loc_name)(_x_)

    n_scale_dim = (tfpl.MultivariateNormalTriL.params_size(ndim) - ndim) if offdiag\
        else (tfpl.IndependentNormal.params_size(ndim) - ndim)
    _scale = tfkl.Dense(n_scale_dim, name=scale_name)(_x_)
    _scale = tf.math.softplus(_scale + _scale_shift) + 1e-5
    if offdiag:
        _scale = tfb.FillTriangular()(_scale)
        make_dist_fn = lambda t: tfd.MultivariateNormalTriL(loc=t[0], scale_tril=t[1])
    else:
        if use_mvn_diag:  # Match type with prior
            make_dist_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1])
        else:
            make_dist_fn = lambda t: tfd.Independent(tfd.Normal(loc=t[0], scale=t[1]))
    return make_dist_fn, [_loc, _scale]


def make_variational(_x, dist_dim,
                     init_std=1.0, offdiag=False,
                     loc_name="loc", scale_name="scale",
                     dist_name="q",
                     use_mvn_diag=True):
    make_dist_fn, dist_params = make_mvn_dist_fn(_x, dist_dim, shift_std=init_std)  #,
                                                #offdiag=offdiag, loc_name=loc_name, scale_name=scale_name,
                                                #use_mvn_diag=use_mvn_diag)
    q_dist = tfpl.DistributionLambda(make_distribution_fn=make_dist_fn,
                                     #convert_to_tensor_fn=lambda s: s.sample(n_samples),
                                     )(dist_params)
    return q_dist


class LearnableMultivariateNormalDiag(tf.keras.Model):
    """Learnable multivariate diagonal normal distribution.

    The model is a multivariate normal distribution with learnable
    `mean` and `stddev` parameters.

    See make_mvn_prior for a description.
    """

    def __init__(self, dimensions, init_std=1.0, trainable_mean=True, trainable_var=True):
        """Constructs a learnable multivariate diagonal normal model.

        Args:
          dimensions: An integer corresponding to the dimensionality of the
            distribution.
        """
        super(LearnableMultivariateNormalDiag, self).__init__()

        with tf.name_scope(self._name):
            self.dimensions = dimensions
            if trainable_mean:
                self._mean = tf.Variable(tf.random.normal([dimensions], stddev=0.1), name="mean")
            else:
                self._mean = tf.zeros(dimensions)
            if trainable_var:
                _scale_shift = np.log(np.exp(init_std) - 1).astype(np.float32)
                self._scale = tfp.util.TransformedVariable(
                    tf.random.normal([dimensions], mean=init_std, stddev=init_std/10, dtype=tf.float32),
                    bijector=tfb.Chain([tfb.Shift(1e-5), tfb.Softplus(), tfb.Shift(_scale_shift)]),
                    name="transformed_scale")
            else:
                self._scale = init_std * tf.ones(dimensions)

    def __call__(self, *args, **kwargs):
        # Allow this Model to be called without inputs.
        dummy = tf.zeros(self.dimensions)
        return super(LearnableMultivariateNormalDiag, self).__call__(
            dummy, *args, **kwargs)

    def call(self, inputs):
        """Runs the model to generate multivariate normal distribution.

        Args:
          inputs: Unused.

        Returns:
          A MultivariateNormalDiag distribution with event shape
          [dimensions], batch shape [], and sample shape [sample_shape,
          dimensions].
        """
        del inputs  # unused
        with tf.name_scope(self._name):
            return tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)

    @property
    def loc(self):
        """The mean of the normal distribution."""
        return self._mean

    @property
    def scale_diag(self):
        """The diagonal standard deviation of the normal distribution."""
        return self._scale


class LearnableMultivariateNormalDiagCell(tf.keras.Model):
    """Multivariate diagonal normal distribution RNN cell.

    The model is an LSTM-based recurrent function that computes the
    parameters for a multivariate normal distribution at each timestep
    `t`.
    """

    def __init__(self, units, output_dimensions, cell_type='lstm'):
        """Constructs a learnable multivariate diagonal normal cell.

        Args:
          units: Dimensionality of the LSTM function parameters.
          output_dimensions: An integer corresponding to the dimensionality of the
            distribution.

        """
        super(LearnableMultivariateNormalDiagCell, self).__init__()
        self.output_dimensions = output_dimensions
        self.units = units
        if cell_type == 'lstm':
            self.rnn_cell = tfkl.LSTMCell(self.units, implementation=1, name="mvndiagcell")
            # why does the jupyter notebook version require implementation=1 but not in pycharm?
        else:
            self.rnn_cell = tfkl.GRUCell(self.units, name="mvndiagcell")
        self.loc_layer = tfkl.Dense(self.output_dimensions, name="mvndiagcell_loc")
        self.scale_untransformed_layer = tfkl.Dense(self.output_dimensions, name="mvndiagcell_scale")

    #def build(self, input_shape):
        #super(LearnableMultivariateNormalDiagCell, self).build(input_shape)
        #self.lstm_cell.build(input_shape)
        #self.loc_layer.build(input_shape)
        #self.scale_untransformed_layer.build(input_shape)
        #self.built = True

    def zero_state(self, sample_batch_shape=()):
        """Returns an initial state for the LSTM cell.

        Args:
          sample_batch_shape: A 0D or 1D tensor of the combined sample and
            batch shape.

        Returns:
          A tuple of the initial previous output at timestep 0 of shape
          [sample_batch_shape, dimensions], and the cell state.
        """
        h0 = tf.zeros([1, self.units])
        c0 = tf.zeros([1, self.units])
        sample_batch_shape = tf.convert_to_tensor(value=sample_batch_shape, dtype=tf.int32)
        out_shape = tf.concat((sample_batch_shape, [self.output_dimensions]), axis=-1)
        previous_output = tf.zeros(out_shape)
        return previous_output, (h0, c0)

    def call(self, inputs, state):
        """Runs the model to generate a distribution for a single timestep.

        This generates a batched MultivariateNormalDiag distribution using
        the output of the recurrent model at the current timestep to
        parameterize the distribution.

        Args:
          inputs: The sampled value of `z` at the previous timestep, i.e.,
            `z_{t-1}`, of shape [..., dimensions].
            `z_0` should be set to the empty matrix.
          state: A tuple containing the (hidden, cell) state.

        Returns:
          A tuple of a MultivariateNormalDiag distribution, and the state of
          the recurrent function at the end of the current timestep. The
          distribution will have event shape [dimensions], batch shape
          [...], and sample shape [sample_shape, ..., dimensions].
        """
        # In order to allow the user to pass in a single example without a batch
        # dimension, we always expand the input to at least two dimensions, then
        # fix the output shape to remove the batch dimension if necessary.
        original_shape = inputs.shape
        if len(original_shape) < 2:
            inputs = tf.reshape(inputs, [1, -1])
        out, state = self.rnn_cell(inputs, state)
        parms_shape = tf.concat((original_shape[:-1], [self.output_dimensions]), 0)
        loc = tf.reshape(self.loc_layer(out), parms_shape)
        scale_diag = self.scale_untransformed_layer(out)
        scale_diag = tf.nn.softplus(scale_diag + scale_shift) + 1e-5
        scale_diag = tf.reshape(scale_diag, parms_shape)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag), state

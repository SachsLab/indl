from typing import Union, Optional, Callable, Tuple, List
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors


def make_learnable_mvn_params(ndim: int, init_std: float = 1.0, trainable_mean: bool = True, trainable_var: bool = True,
                              offdiag: bool = False):
    """
    Return mean (loc) and stddev (scale) parameters for initializing multivariate normal distributions.
    If trainable_mean then it will be initialized with random normal (stddev=0.1), otherwise zeros.
    If trainable_var then it will be initialized with random normal centered at a value such that the
        bijector transformation yields the value in init_std. When init_std is 1.0 (default) then the
        inverse-bijected value is approximately 0.0.
        If not trainable_var then scale is a vector or matrix of init_std of appropriate shape for the dist.
    Args:
        ndim: Number of dimensions.
        init_std: Initial value for the standard deviation.
        trainable_mean: Whether or not the mean (loc) is a trainable tf.Variable.
        trainable_var: Whether or not the variance (scale) is a trainable tf.Variable.
        offdiag: Whether or not off-diagonal elements are allowed.

    Returns: loc, scale
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

    return loc, scale


def make_mvn_prior(ndim: int, init_std: float = 1.0, trainable_mean: bool = True, trainable_var: bool = True,
                   offdiag: bool = False) -> Union[tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL]:
    """
    Creates a tensorflow-probability distribution:
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
    loc, scale = make_learnable_mvn_params(ndim, init_std=init_std,
                                           trainable_mean=trainable_mean, trainable_var=trainable_var,
                                           offdiag=offdiag)

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


def make_mvn_dist_fn(_x_: tf.Tensor, ndim: int, shift_std: float = 1.0, offdiag: bool = False,
                     loc_name: Optional[str] = None, scale_name: Optional[str] = None, use_mvn_diag: bool = True
                     ) -> Tuple[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution], List[tf.Tensor]]:
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
        make_dist_fn, [loc, scale]
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


def make_variational(x: tf.Tensor, dist_dim: int,
                     init_std: float = 1.0, offdiag: bool = False,
                     samps: int = 1,
                     loc_name="loc", scale_name="scale",
                     dist_name="q",
                     use_mvn_diag: bool = True
                     ) -> Union[tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL, tfd.Independent]:
    """
    Take an input tensor and return a multivariate normal distribution parameterized by that input tensor.

    Args:
        x: input tensor
        dist_dim: the dimensionality of the distribution
        init_std: initial stddev SHIFT of the distribution when input is 0.
        offdiag: whether or not to include covariances
        samps: the number of samples to draw when using implied convert_to_tensor_fn
        loc_name: not used (I need to handle naming better)
        scale_name: not used
        dist_name: not used
        use_mvn_diag: whether to use tfd.MultivariateNormal(Diag|TriL) (True) or tfd.Independent(tfd.Normal) (False)
            Latter is untested. Note that the mvn dists will put the timesteps dimension (if present in the input)
            into the "batch dimension" while the "event" dimension will be the last dimension only.
            You can use tfd.Independent(q_dist, reinterpreted_batch_ndims=1) to move the timestep dimension to the
            event dimension if necessary. (tfd.Independent doesn't play well with tf.keras.Model inputs/outputs).

    Returns:
        A tfd.Distribution. The distribution is of type MultivariateNormalDiag (or MultivariateNormalTriL if offdiag)
        if use_mvn_diag is set.
    """
    make_dist_fn, dist_params = make_mvn_dist_fn(x, dist_dim, shift_std=init_std,
                                                 offdiag=offdiag,
                                                 # loc_name=loc_name, scale_name=scale_name,
                                                 use_mvn_diag=use_mvn_diag)
    # Python `callable` that takes a `tfd.Distribution`
    #         instance and returns a `tf.Tensor`-like object.

    """
    # Unfortunately I couldn't get this to work :(
    # Will have to explicitly q_f.value() | qf.mean() from dist in train_step
    def custom_convert_fn(d, training=None):
        if training is None:
            training = K.learning_phase()
        output = tf_utils.smart_cond(training,
                                     lambda: d.sample(samps),
                                     lambda: d.mean()
                                     )
        return output
    def convert_fn(d):
        return K.in_train_phase(tfd.Distribution.sample if samps <= 1 else lambda: d.sample(samps),
        lambda: d.mean())
    """
    convert_fn = tfd.Distribution.sample if samps <= 1 else lambda d: d.sample(samps)
    q_dist = tfpl.DistributionLambda(make_distribution_fn=make_dist_fn,
                                     convert_to_tensor_fn=convert_fn,
                                     )(dist_params)
    # if tf.shape(x).shape[0] > 2:
    #     q_dist = tfd.Independent(q_dist, reinterpreted_batch_ndims=1)
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


class LearnableMultivariateNormalCell(tf.keras.Model):
    """Multivariate normal distribution RNN cell.

    The model is a RNN-based recurrent function that computes the
    parameters for a multivariate normal distribution at each timestep `t`.
    Based on:
    https://github.com/tensorflow/probability/blob/698e0101aecf46c42858db7952ee3024e091c291/tensorflow_probability/examples/disentangled_vae.py#L242
    """

    def __init__(self, units: int, out_dim: int,
                 shift_std: float = 0.1, cell_type: str = 'lstm', offdiag: bool = False):
        """Constructs a learnable multivariate normal cell.

        Args:
          units: Dimensionality of the RNN function parameters.
          out_dim: The dimensionality of the distribution.
          shift_std: Shift applied to MVN std before building the dist. Providing a shift
            toward the expected std allows the input values to be closer to 0.
          cell_type: an RNN cell type among 'lstm', 'gru', 'rnn', 'gruclip'. case-insensitive.
          offdiag: set True to allow non-zero covariance (within-timestep) in the returned distribution.
        """
        super(LearnableMultivariateNormalCell, self).__init__()
        self.offdiag = offdiag
        self.output_dimensions = out_dim
        self.units = units
        if cell_type.upper().endswith('LSTM'):
            self.rnn_cell = tfkl.LSTMCell(self.units, implementation=1, name="mvncell")
            # why does the jupyter notebook version require implementation=1 but not in pycharm?
        elif cell_type.upper().endswith('GRU'):
            self.rnn_cell = tfkl.GRUCell(self.units, name="mvnell")
        elif cell_type.upper().endswith('RNN'):
            self.rnn_cell = tfkl.SimpleRNNCell(self.units, name="mvncell")
        elif cell_type.upper().endswith('GRUCLIP'):
            from indl.rnn.gru_clip import GRUClipCell
            self.rnn_cell = GRUClipCell(self.units, name="mvncell")
        else:
            raise ValueError("cell_type %s not recognized" % cell_type)

        self.loc_layer = tfkl.Dense(self.output_dimensions, name="mvncell_loc")
        n_scale_dim = (tfpl.MultivariateNormalTriL.params_size(out_dim) - out_dim) if offdiag\
            else (tfpl.IndependentNormal.params_size(out_dim) - out_dim)
        self.scale_untransformed_layer = tfkl.Dense(n_scale_dim, name="mvndiagcell_scale")
        self._scale_shift = np.log(np.exp(shift_std) - 1).astype(np.float32)

    #def build(self, input_shape):
        #super(LearnableMultivariateNormalDiagCell, self).build(input_shape)
        #self.lstm_cell.build(input_shape)
        #self.loc_layer.build(input_shape)
        #self.scale_untransformed_layer.build(input_shape)
        #self.built = True

    def zero_state(self, sample_batch_shape=()):
        """Returns an initial state for the RNN cell.

        Args:
          sample_batch_shape: A 0D or 1D tensor of the combined sample and
            batch shape.

        Returns:
          A tuple of the initial previous output at timestep 0 of shape
          [sample_batch_shape, dimensions], and the cell state.
        """
        zero_state = self.rnn_cell.get_initial_state(batch_size=sample_batch_shape[-1], dtype=tf.float32)
        sample_batch_shape = tf.convert_to_tensor(value=sample_batch_shape, dtype=tf.int32)
        out_shape = tf.concat((sample_batch_shape, [self.output_dimensions]), axis=-1)
        previous_output = tf.zeros(out_shape)
        return previous_output, zero_state

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
        scale = self.scale_untransformed_layer(out)
        scale = tf.nn.softplus(scale + self._scale_shift) + 1e-5
        scale = tf.reshape(scale, parms_shape)
        if self.offdiag:
            return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale)
        else:
            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), state

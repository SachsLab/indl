__all__ = ['diag_gaussian_log_likelihood', 'gaussian_pos_log_likelihood', 'Gaussian',
           'DiagonalGaussianFromExisting', 'LearnableDiagonalGaussian', 'LearnableAutoRegressive1Prior',
           'CoordinatedDropout', 'GRUClipCell', 'test_GRUClipCell', 'GRUClip', 'test_GRUClipLayer',
           'test_GRUClipLayer_in_Bidirectional']


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.eager import context
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import GRU as GRUv1
tfd = tfp.distributions
tfpl = tfp.layers


def diag_gaussian_log_likelihood(z, mu=0.0, logvar=0.0):
    """Log-likelihood under a Gaussian distribution with diagonal covariance.
      Returns the log-likelihood for each dimension.  One should sum the
      results for the log-likelihood under the full multidimensional model.

    Args:
      z: The value to compute the log-likelihood.
      mu: The mean of the Gaussian
      logvar: The log variance of the Gaussian.

    Returns:
      The log-likelihood under the Gaussian model.
    """

    return -0.5 * (logvar + np.log(2 * np.pi) + \
                   tf.square((z - mu) / tf.exp(0.5 * logvar)))


def gaussian_pos_log_likelihood(unused_mean, logvar, noise):
    """Gaussian log-likelihood function for a posterior in VAE

    Note: This function is specialized for a posterior distribution, that has the
    form of z = mean + sigma * noise.

    Args:
        unused_mean: ignore
        logvar: The log variance of the distribution
        noise: The noise used in the sampling of the posterior.

    Returns:
        The log-likelihood under the Gaussian model.
    """
    # ln N(z; mean, sigma) = - ln(sigma) - 0.5 ln 2pi - noise^2 / 2
    return - 0.5 * (logvar + np.log(2 * np.pi) + tf.square(noise))


class Gaussian(object):
    """Base class for Gaussian distribution classes."""
    @property
    def mean(self):
        return self.mean_bxn

    @property
    def logvar(self):
        return self.logvar_bxn

    @property
    def noise(self):
        return tf.random.normal(tf.shape(input=self.logvar))

    @property
    def sample(self):
        # return self.mean + tf.exp(0.5 * self.logvar) * self.noise
        return self.sample_bxn


class DiagonalGaussianFromExisting(Gaussian):
    """
    Diagonal Gaussian with different constant mean and variances in each
    dimension.
    """

    def __init__(self, mean_bxn, logvar_bxn, var_min=0.0):
        self.mean_bxn = mean_bxn
        if var_min > 0.0:
            logvar_bxn = tf.math.log(tf.exp(logvar_bxn) + var_min)
            # logvar_bxn = tf.nn.relu(logvar_bxn) + tf.math.log(var_min)
        self.logvar_bxn = logvar_bxn

        self.noise_bxn = noise_bxn = tf.random.normal(tf.shape(input=logvar_bxn))
        #self.noise_bxn.set_shape([None, z_size])
        self.sample_bxn = mean_bxn + tf.exp(0.5 * logvar_bxn) * noise_bxn

    def logp(self, z=None):
        """Compute the log-likelihood under the distribution.

        Args:
          z (optional): value to compute likelihood for, if None, use sample.

        Returns:
          The likelihood of z under the model.
        """
        if z is None:
            z = self.sample

        # This is needed to make sure that the gradients are simple.
        # The value of the function shouldn't change.
        if z == self.sample_bxn:
            return gaussian_pos_log_likelihood(self.mean_bxn, self.logvar_bxn, self.noise_bxn)

        return diag_gaussian_log_likelihood(z, self.mean_bxn, self.logvar_bxn)


class LearnableDiagonalGaussian(Gaussian):
    """
    Diagonal Gaussian with different means and variances in each
    dimension. Means and variances are optionally trainable.
    For LFADS ics prior, trainable_mean=True, trainable_var=False (both default).
    For LFADS cos prior (if not using AR1), trainable_mean=False, trainable_var=True
    """

    def __init__(self, batch_size, z_size, name, var, trainable_mean=True, trainable_var=False):
        # MRK's fix, letting the mean of the prior to be trainable
        mean_init = 0.0
        num_steps = z_size[0]
        num_dim = z_size[1]
        z_mean_1xn = tf.compat.v1.get_variable(name=name+"/mean", shape=[1,1,num_dim],
                                               initializer=tf.compat.v1.constant_initializer(mean_init),
                                               trainable=trainable_mean)
        self.mean_bxn = tf.tile(z_mean_1xn, tf.stack([batch_size, num_steps, 1]))
        self.mean_bxn.set_shape([None] + z_size)

        # MRK, make Var trainable (for Controller prior)
        var_init = np.log(var)
        z_logvar_1xn = tf.compat.v1.get_variable(name=name+"/logvar", shape=[1,1,num_dim],
                                       initializer=tf.compat.v1.constant_initializer(var_init),
                                       trainable=trainable_var)
        self.logvar_bxn = tf.tile(z_logvar_1xn, tf.stack([batch_size, num_steps, 1]))
        self.logvar_bxn.set_shape([None] + z_size)
        # remove time axis if 1 (used for ICs)
        if num_steps == 1:
            self.mean_bxn = tf.squeeze(self.mean_bxn, axis=1)
            self.logvar_bxn = tf.squeeze(self.logvar_bxn, axis=1)

        self.noise_bxn = tf.random.normal(tf.shape(input=self.logvar_bxn))


class LearnableAutoRegressive1Prior(object):
    """
    AR(1) model where autocorrelation and process variance are learned
    parameters.  Assumed zero mean.
    """

    def __init__(self, batch_size, z_size,
               autocorrelation_taus, noise_variances,
               do_train_prior_ar_atau, do_train_prior_ar_nvar,
               name):
        """Create a learnable autoregressive (1) process.

        Args:
          batch_size: The size of the batch, i.e. 0th dim in 2D tensor of samples.
          z_size: The dimension of the distribution, i.e. 1st dim in 2D tensor.
          autocorrelation_taus: The auto correlation time constant of the AR(1)
          process.
            A value of 0 is uncorrelated gaussian noise.
          noise_variances: The variance of the additive noise, *not* the process
            variance.
          do_train_prior_ar_atau: Train or leave as constant, the autocorrelation?
          do_train_prior_ar_nvar: Train or leave as constant, the noise variance?
          num_steps: Number of steps to run the process.
          name: The name to prefix to learned TF variables.
        """

        # Note the use of the plural in all of these quantities.  This is intended
        # to mark that even though a sample z_t from the posterior is thought of a
        # single sample of a multidimensional gaussian, the prior is actually
        # thought of as U AR(1) processes, where U is the dimension of the inferred
        # input.
        size_bx1 = tf.stack([batch_size, 1])
        size__xu = [None, z_size]
        # process variance, the variance at time t over all instantiations of AR(1)
        # with these parameters.
        log_evar_inits_1xu = tf.expand_dims(tf.math.log(noise_variances), 0)
        self.logevars_1xu = logevars_1xu = \
            tf.Variable(log_evar_inits_1xu, name=name+"/logevars", dtype=tf.float32,
                        trainable=do_train_prior_ar_nvar)
        self.logevars_bxu = logevars_bxu = tf.tile(logevars_1xu, size_bx1)
        logevars_bxu.set_shape(size__xu) # tile loses shape

        # \tau, which is the autocorrelation time constant of the AR(1) process
        log_atau_inits_1xu = tf.expand_dims(tf.math.log(autocorrelation_taus), 0)
        self.logataus_1xu = logataus_1xu = \
            tf.Variable(log_atau_inits_1xu, name=name+"/logatau", dtype=tf.float32,
                        trainable=do_train_prior_ar_atau)

        # phi in x_t = \mu + phi x_tm1 + \eps
        # phi = exp(-1/tau)
        # phi = exp(-1/exp(logtau))
        # phi = exp(-exp(-logtau))
        phis_1xu = tf.exp(-tf.exp(-logataus_1xu))
        self.phis_bxu = phis_bxu = tf.tile(phis_1xu, size_bx1)
        phis_bxu.set_shape(size__xu)

        # process noise
        # pvar = evar / (1- phi^2)
        # logpvar = log ( exp(logevar) / (1 - phi^2) )
        # logpvar = logevar - log(1-phi^2)
        # logpvar = logevar - (log(1-phi) + log(1+phi))
        self.logpvars_1xu = \
            logevars_1xu - tf.math.log(1.0-phis_1xu) - tf.math.log(1.0+phis_1xu)
        self.logpvars_bxu = logpvars_bxu = tf.tile(self.logpvars_1xu, size_bx1)
        logpvars_bxu.set_shape(size__xu)

        # process mean (zero but included in for completeness)
        self.pmeans_bxu = pmeans_bxu = tf.zeros_like(phis_bxu)


    def logp_t(self, z_t_bxu, z_tm1_bxu=None):
        """Compute the log-likelihood under the distribution for a given time t,
        not the whole sequence.

        Args:
          z_t_bxu: sample to compute likelihood for at time t.
          z_tm1_bxu (optional): sample condition probability of z_t upon.

        Returns:
          The likelihood of p_t under the model at time t. i.e.
            p(z_t|z_tm1_bxu) = N(z_tm1_bxu * phis, eps^2)

        """
        if z_tm1_bxu is None:
            logp_tgtm1_bxu = diag_gaussian_log_likelihood(z_t_bxu, self.pmeans_bxu, self.logpvars_bxu)
        else:
            means_t_bxu = self.pmeans_bxu + self.phis_bxu * z_tm1_bxu
            logp_tgtm1_bxu = diag_gaussian_log_likelihood(z_t_bxu, means_t_bxu, self.logevars_bxu)
        return logp_tgtm1_bxu


class CoordinatedDropout(tfkl.Dropout):
    def compute_output_shape(self, input_shape):
        return input_shape, input_shape

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            rate = self.rate
            noise_shape = self.noise_shape
            seed = self.seed
            name = None
            with ops.name_scope(None, "coordinated_dropout", [inputs]) as name:
                is_rate_number = isinstance(rate, numbers.Real)
                if is_rate_number and (rate < 0 or rate >= 1):
                    raise ValueError("rate must be a scalar tensor or a float in the "
                                     "range [0, 1), got %g" % rate)
                x = ops.convert_to_tensor(inputs, name="x")
                x_dtype = x.dtype
                if not x_dtype.is_floating:
                    raise ValueError("x has to be a floating point tensor since it's going "
                                     "to be scaled. Got a %s tensor instead." % x_dtype)
                is_executing_eagerly = context.executing_eagerly()
                if not tensor_util.is_tensor(rate):
                    if is_rate_number:
                        keep_prob = 1 - rate
                        scale = 1 / keep_prob
                        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
                        ret = gen_math_ops.mul(x, scale)
                    else:
                        raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)
                else:
                    rate.get_shape().assert_has_rank(0)
                    rate_dtype = rate.dtype
                    if rate_dtype != x_dtype:
                        if not rate_dtype.is_compatible_with(x_dtype):
                            raise ValueError(
                                "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
                                (x_dtype.name, rate_dtype.name, rate))
                        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
                    one_tensor = constant_op.constant(1, dtype=x_dtype)
                    ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

                noise_shape = nn_ops._get_noise_shape(x, noise_shape)
                # Sample a uniform distribution on [0.0, 1.0) and select values larger
                # than rate.
                #
                # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
                # and subtract 1.0.
                random_tensor = random_ops.random_uniform(
                    noise_shape, seed=seed, dtype=x_dtype)
                # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
                # hence a >= comparison is used.
                keep_mask = random_tensor >= rate
                ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
                if not is_executing_eagerly:
                    ret.set_shape(x.get_shape())
                return (ret, keep_mask)

        output = tf_utils.smart_cond(training,
                                     dropped_inputs,
                                     lambda: (array_ops.identity(inputs), array_ops.ones_like(inputs) > 0))
        return output


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


def test_GRUClipCell():
    K.clear_session()

    n_times, n_sensors = 246, 36
    batch_size = 16
    f_units = 128

    f_enc_inputs = tf.keras.Input(shape=(n_times, n_sensors))
    cell = GRUClipCell(f_units)
    rnn = tfkl.RNN(cell)
    bidir = tfkl.Bidirectional(rnn)
    final_state = bidir(f_enc_inputs)

    model = tf.keras.Model(inputs=f_enc_inputs, outputs=final_state, name="GRUClip")
    model.summary()

    dummy_state = model(tf.random.uniform((batch_size, n_times, n_sensors)))
    print(dummy_state)


class GRUClip(GRUv1):
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


def test_GRUClipLayer():
    K.clear_session()

    n_times, n_sensors = 246, 36
    batch_size = 16
    f_units = 128

    f_enc_inputs = tf.keras.Input(shape=(n_times, n_sensors))
    final_state = GRUClip(f_units)(f_enc_inputs)

    model = tf.keras.Model(inputs=f_enc_inputs, outputs=final_state, name="GRUClip")
    model.summary()

    dummy_state = model(tf.random.uniform((batch_size, n_times, n_sensors)))
    print(dummy_state)


def test_GRUClipLayer_in_Bidirectional():
    K.clear_session()

    n_times, n_sensors = 246, 36
    batch_size = 16
    f_units = 128

    f_enc_inputs = tf.keras.Input(shape=(n_times, n_sensors))

    rnn_layer = GRUClip(f_units)

    bd_layer = tfkl.Bidirectional(rnn_layer, merge_mode="concat")
    final_state = bd_layer(f_enc_inputs)

    model = tf.keras.Model(inputs=f_enc_inputs, outputs=final_state, name="GRUClipBi")
    model.summary()

    dummy_state = model(tf.random.uniform((batch_size, n_times, n_sensors)))
    print(dummy_state)


def main():
    #TODO: test_DiagonalGaussianFromExisting
    #TODO: test_CoordinatedDropout
    test_GRUClipCell()
    test_GRUClipLayer()
    test_GRUClipLayer_in_Bidirectional()


if __name__ == "__main__":
    main()

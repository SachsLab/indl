__all__ = ['diag_gaussian_log_likelihood', 'gaussian_pos_log_likelihood', 'Gaussian',
           'DiagonalGaussianFromExisting', 'LearnableDiagonalGaussian', 'LearnableAutoRegressive1Prior']


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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
        z_mean_1xn = tf.compat.v1.get_variable(name=name+"/mean", shape=[1, 1, num_dim],
                                               initializer=tf.compat.v1.constant_initializer(mean_init),
                                               trainable=trainable_mean)
        self.mean_bxn = tf.tile(z_mean_1xn, tf.stack([batch_size, num_steps, 1]))
        self.mean_bxn.set_shape([None] + z_size)

        # MRK, make Var trainable (for Controller prior)
        var_init = np.log(var)
        z_logvar_1xn = tf.compat.v1.get_variable(name=name+"/logvar", shape=[1, 1, num_dim],
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
        logevars_bxu.set_shape(size__xu)  # tile loses shape

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


# Used for AR prior
class KLCost_GaussianGaussianProcessSampled(object):
    """ log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian process
    prior via sampling.

    The log p(x|z) term is the reconstruction error under the model.
    The KL term represents the penalty for passing information from the encoder
    to the decoder.
    To sample KL(q||p), we simply sample
        ln q - ln p
    by drawing samples from q and averaging.
    """

    def __init__(self, post_zs, prior_z_process):
        """Create a lower bound in three parts, normalized reconstruction
        cost, normalized KL divergence cost, and their sum.

        Args:
          post_zs: posterior z ~ q(z|x)
          prior_z_process: prior AR(1) process
        """
        # assert len(post_zs) > 1, "GP is for time, need more than 1 time step."
        # assert isinstance(prior_z_process, GaussianProcess), "Must use GP."

        # L = -KL + log p(x|z), to maximize bound on likelihood
        # -L = KL - log p(x|z), to minimize bound on NLL
        # so 'KL cost' is postive KL divergence

        # sample from the posterior for all time points and dimensions
        post_zs_sampled = post_zs.sample
        # sum KL over time and dimension axis
        logq_bxu = tf.reduce_sum(input_tensor=post_zs.logp(post_zs_sampled), axis=[1, 2])

        logp_bxu = 0
        num_steps = post_zs.mean.get_shape()[1]
        for i in range(num_steps):
            # posterior is independent in time, prior is not
            if i == 0:
                z_tm1_bxu = None
            else:
                z_tm1_bxu = post_zs_sampled[:, i-1, :]
            logp_bxu += tf.reduce_sum(input_tensor=prior_z_process.logp_t(
                post_zs_sampled[:, i, :], z_tm1_bxu), axis=[1])

        kl_b = logq_bxu - logp_bxu
        self.kl_cost_b = kl_b

import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
scale_shift = np.log(np.exp(1) - 1).astype(np.float32)


def test_make_mvn_prior():
    from indl.model.tfp import make_mvn_prior

    def _test(latent_size=5, init_std=0.1, trainable_mean=True, trainable_var=True, offdiag=False):
        prior = make_mvn_prior(latent_size, init_std=init_std,
                               trainable_mean=trainable_mean, trainable_var=trainable_var,
                               offdiag=offdiag)
        assert (isinstance(prior.loc, tf.Variable) == trainable_mean)
        if offdiag:
            assert (hasattr(prior.scale_tril, 'trainable_variables') == trainable_var)
        else:
            assert ((len(prior.scale.trainable_variables) > 0) == trainable_var)
        if not trainable_var:
            assert np.all(prior.stddev().numpy() == init_std)
        if not trainable_mean:
            assert np.all(prior.mean().numpy() == 0.0)

    for _mean in True, False:
        for _var in True, False:
            for _offd in True, False:
                _test(trainable_mean=_mean, trainable_var=_var, offdiag=_offd)


def _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size):
    assert isinstance(q_dist, tfd.MultivariateNormalDiag)

    # Test in a model with a data tensor
    model = tf.keras.Model(inputs=inputs, outputs=q_dist)
    dummy_inputs = tf.random.uniform((batch_size, input_dim))
    dummy_q = model(dummy_inputs)
    assert isinstance(dummy_q, tfd.MultivariateNormalDiag)
    assert dummy_q.stddev().shape.as_list() == [batch_size, dist_dim]
    assert np.all(dummy_q.stddev().numpy() > 0)
    assert dummy_q.sample().shape.as_list() == [batch_size, dist_dim]
    assert ~np.any(np.isnan(dummy_q.sample().numpy()))


def test_make_mvn_dist_fn():
    from indl.model.tfp import make_mvn_dist_fn

    input_dim = 4
    dist_dim = 3
    batch_size = 8

    # Test with placeholder
    inputs = tfkl.Input(shape=(input_dim,))
    # First the callable
    make_dist_fn, dist_params = make_mvn_dist_fn(inputs, dist_dim, shift_std=0.1)
    assert hasattr(make_dist_fn, '__call__')
    assert isinstance(dist_params[0], tf.Tensor)
    assert isinstance(dist_params[1], tf.Tensor)
    # Then test using it to make a distribution
    q_dist = tfpl.DistributionLambda(make_distribution_fn=make_dist_fn,
                                     # convert_to_tensor_fn=lambda s: s.sample(n_samples),
                                     )(dist_params)
    _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size)


def test_make_variational():
    from indl.model.tfp import make_variational

    input_dim = 4
    dist_dim = 3
    batch_size = 8

    # Test making a placeholder variational.
    inputs = tfkl.Input(shape=(input_dim,))
    q_dist = make_variational(inputs, dist_dim, init_std=0.1)
    _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size)

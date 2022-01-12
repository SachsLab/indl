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
    from indl.dists import make_mvn_prior

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
                K.clear_session()
                _test(trainable_mean=_mean, trainable_var=_var, offdiag=_offd)


def _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size):
    # assert isinstance(q_dist, (tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL))
    assert K.is_keras_tensor(q_dist)

    # Test in a model with a data tensor
    model = tf.keras.Model(inputs=inputs, outputs=q_dist)
    dummy_inputs = tf.random.uniform((batch_size, input_dim))
    dummy_q = model(dummy_inputs)
    assert isinstance(dummy_q, (tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL))
    assert dummy_q.stddev().shape.as_list() == [batch_size, dist_dim]
    assert np.all(dummy_q.stddev().numpy() > 0)
    assert dummy_q.sample().shape.as_list() == [batch_size, dist_dim]
    assert ~np.any(np.isnan(dummy_q.sample().numpy()))
    # Implied convert_to_tensor != mean
    assert tf.not_equal(tf.add(dummy_q, 0), dummy_q.mean()).numpy().any()

    # Test implied tensor - doesn't work.
    # K.set_learning_phase(0)  # Default
    # tf.assert_equal(tf.add(dummy_q, 0), dummy_q.mean())
    # K.set_learning_phase(1)
    # assert tf.not_equal(tf.add(dummy_q, 0), dummy_q.mean()).numpy().any()


def test_make_mvn_dist_fn():
    from indl.dists import make_mvn_dist_fn

    input_dim = 4
    dist_dim = 3
    batch_size = 8

    # Test with placeholder
    inputs = tfkl.Input(shape=(input_dim,))
    # First the callable
    make_dist_fn, dist_params = make_mvn_dist_fn(inputs, dist_dim, shift_std=0.1)
    assert hasattr(make_dist_fn, '__call__')
    # tensorflow.python.keras.engine.keras_tensor.KerasTensor
    # assert isinstance(dist_params[0], tf.Tensor)
    # assert isinstance(dist_params[1], tf.Tensor)
    assert K.is_keras_tensor(dist_params[0])
    assert K.is_keras_tensor(dist_params[1])
    test_dist = make_dist_fn((dist_params[0], dist_params[1]))
    assert isinstance(test_dist, tfd.Distribution)
    # Then test using it to make a distribution
    _dist_layer = tfpl.DistributionLambda(make_distribution_fn=make_dist_fn,
                                          # convert_to_tensor_fn=lambda s: s.sample(n_samples),
                                          )
    q_dist = _dist_layer(dist_params)
    _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size)


def test_make_variational():
    from indl.dists import make_variational

    input_dim = 4
    dist_dim = 3
    batch_size = 8

    # Test making a placeholder variational.
    K.clear_session()
    inputs = tfkl.Input(shape=(input_dim,))

    q_dist = make_variational(inputs, dist_dim, init_std=0.1, offdiag=False, samps=2)
    assert(isinstance(q_dist, tfd.MultivariateNormalDiag))
    _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size)

    q_dist = make_variational(inputs, dist_dim, init_std=0.1, offdiag=True, samps=1)
    assert (isinstance(q_dist, tfd.MultivariateNormalTriL))
    _run_assertions_on_qdist(q_dist, inputs, input_dim, dist_dim, batch_size)

    # With more than 1 samp
    n_samps = 5
    q_dist = make_variational(inputs, dist_dim, init_std=0.1, offdiag=False, samps=n_samps)
    model = tf.keras.Model(inputs=inputs, outputs=q_dist)
    dummy_inputs = tf.random.uniform((batch_size, input_dim))
    dummy_q = model(dummy_inputs)
    # By requiring a tensor without explicitly calling .sample() or .mean(),
    #  we invoke the convert_to_tensor_fn parameterized with samps
    assert tf.add(dummy_q, 0).shape.as_list() == [n_samps, batch_size, dist_dim]

    # With a timestep dimension.
    K.clear_session()
    timesteps = 10
    inputs = tfkl.Input(shape=(timesteps, input_dim))
    q_dist = make_variational(inputs, dist_dim)
    assert isinstance(q_dist, tfd.MultivariateNormalDiag)
    model = tf.keras.Model(inputs=inputs, outputs=q_dist)
    dummy_inputs = tf.random.uniform((batch_size, timesteps, input_dim))
    dummy_q = model(dummy_inputs)
    assert dummy_q.batch_shape.as_list() == [batch_size, timesteps]
    assert dummy_q.event_shape.as_list() == [dist_dim]

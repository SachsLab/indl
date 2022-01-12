import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import distribution as distribution_lib
tfd = tfp.distributions


def _basic_test(dist, sample, latent_dim, timesteps, batch_size):
    assert (isinstance(dist, distribution_lib.Distribution))
    assert (dist.event_shape.as_list() == [timesteps, latent_dim])
    assert (isinstance(sample, tf.Tensor))
    assert (sample.shape.as_list() == [1, batch_size, timesteps, latent_dim])


def test_ar1_mvn_generator():
    from indl.dists.sequential import AR1ProcessMVNGenerator

    latent_dim = 4
    timesteps = 20
    init_taus = 10.0
    init_std = 0.1
    batch_size = 16
    trainable_tau = True
    trainable_var = True
    offdiag = False

    generator = AR1ProcessMVNGenerator([init_taus for _ in range(latent_dim)],
                                       init_std=init_std,
                                       trainable_tau=trainable_tau,
                                       trainable_var=trainable_var,
                                       offdiag=offdiag)
    sample, dist = generator.get_dist(timesteps, samples=1, batch_size=batch_size)
    _basic_test(dist, sample, latent_dim, timesteps, batch_size)
    # _kl_test(dist, latent_dim, offdiag, init_std, timesteps, batch_size, z2_units=16)


def test_rnn_mvn_generator():
    from indl.dists.sequential import RNNMVNGenerator

    latent_dim = 4
    n_units = 12
    timesteps = 20
    batch_size = 8
    init_std = 0.1
    offdiag = False

    for cell_type in ['GRUClip', 'GRU', 'LSTM', 'RNN']:
        generator = RNNMVNGenerator(n_units, latent_dim, cell_type, shift_std=init_std, offdiag=offdiag)
        sample, dist = generator.get_dist(timesteps, samples=1, batch_size=batch_size, fixed=True)
        _basic_test(dist, sample, latent_dim, timesteps, batch_size)

    # _kl_test(dist, latent_dim, offdiag, init_std, timesteps, batch_size)


def test_tiled_mvn_generator():
    from indl.dists.sequential import TiledMVNGenerator

    init_std = 0.1
    latent_dim = 4  # z_latent_size
    batch_size = 8
    timesteps = 20

    for trainable_mean in [False, True]:
        for trainable_var in [False, True]:
            for offdiag in [True, False]:
                generator = TiledMVNGenerator(latent_dim, init_std=init_std,
                                              trainable_mean=trainable_mean,
                                              trainable_var=trainable_var,
                                              offdiag=offdiag)
                # The generator ._loc and ._scale are the result of a call to expand_dims.
                # I don't know how to check if the input to that op is a trainable var.
                sample, dist = generator.get_dist(timesteps, samples=1, batch_size=batch_size)
                _basic_test(dist, sample, latent_dim, timesteps, batch_size)
                if not trainable_var:
                    assert np.all(dist.stddev().numpy() == init_std)
                if not trainable_mean:
                    assert np.all(dist.mean().numpy() == 0.0)

    # _kl_test(dist, latent_dim, offdiag, init_std, timesteps, batch_size)



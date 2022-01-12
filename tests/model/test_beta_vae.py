import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as tfkl
import tensorflow_probability as tfp
tfd = tfp.distributions


def test_prepare_inputs(n_times=None, n_sensors=36):
    from indl.model.beta_vae import prepare_inputs

    K.clear_session()
    # These parameters are arbitrary, just used for testing. They are not to be used in a real model.
    params = {
        'dropout_rate': 0.01,
        'encs_input_samps': 20,
        'coordinated_dropout_rate': 0.1,
        'input_factors': 24
    }
    inputs = tf.keras.Input(shape=(n_times, n_sensors))

    encs_inputs, encd_inputs, cd_mask = prepare_inputs(params, inputs)

    assert len(encs_inputs.shape) == 3 and encs_inputs.shape[-1] == params['input_factors']
    assert len(encd_inputs.shape) == 3 and encd_inputs.shape[-1] == params['input_factors']
    assert len(cd_mask.shape) == 3 and cd_mask.shape[-1] == n_sensors
    # Coordinated dropout has its own unit tests so we can skip that here.
    # Put it into a model and run a full tensor through it.
    test_model = tf.keras.Model(inputs=inputs, outputs=[encs_inputs, encd_inputs, cd_mask])
    batch_size = 16
    n_times = n_times or (params['encs_input_samps'] + 2)
    full_inputs = tf.random.uniform((batch_size, n_times, n_sensors))
    _f, _z, _cd = test_model(full_inputs)
    assert _f.shape == (batch_size, params['encs_input_samps'], params['input_factors'])
    assert _z.shape == (batch_size, n_times, params['input_factors'])

    # Test with encs_input_samps = 0: _f should have full timestamps
    params['encs_input_samps'] = 0
    encs_inputs, encd_inputs, cd_mask = prepare_inputs(params, inputs)
    test_model = tf.keras.Model(inputs=inputs, outputs=[encs_inputs, encd_inputs, cd_mask])
    _f, _z, _cd = test_model(full_inputs)
    assert _f.shape[1] == n_times


def test_create_encs(n_times=None, n_sensors=36):
    from indl.model.beta_vae import create_encs

    K.clear_session()
    params = {
        # First two params aren't needed for create_encs functionality,
        #  but they are needed to construct input of appropriate shape.
        'encs_input_samps': n_times if n_times else 40,
        'input_factors': 24,
        'encs_rnn_units': 12,
        'encs_rnn_type': 'BidirectionalGRUClip',  # 'GRU', 'LSTM', 'SimpleRNN', 'GRUClip', with or without Bidirectional
        'gru_clip_value': 5.0,
    }
    input_samps = params['encs_input_samps'] or n_times
    input_dim = params['input_factors'] or n_sensors
    encs_inputs = tf.keras.Input(shape=(input_samps, input_dim))
    # Run through each RNN type
    for encs_rnn_type in ['GRU', 'LSTM', 'SimpleRNN', 'GRUClip']:
        for bidir_prefix in ['', 'Bidirectional']:
            params['encs_rnn_type'] = bidir_prefix + encs_rnn_type
            encoded_s = create_encs(params, encs_inputs)
            assert len(encoded_s.shape) == 2
            out_dim = (2 if bidir_prefix == 'Bidirectional' else 1) * params['encs_rnn_units']
            assert encoded_s.shape[1] == out_dim

    # Test with multiple layers
    params['encs_rnn_units'] = [12, 18]
    encoded_s = create_encs(params, encs_inputs)
    assert len(encoded_s.shape) == 2
    assert encoded_s.shape[1] == 2 * params['encs_rnn_units'][-1]

    # Now run it through a model
    test_model = tf.keras.Model(inputs=encs_inputs, outputs=encoded_s)
    batch_size = 16
    n_times = n_times or params['encs_input_samps']
    full_inputs = tf.random.uniform((batch_size, n_times, input_dim))
    out = test_model(full_inputs)
    assert out.shape[1] == 2 * params['encs_rnn_units'][-1]


def test_make_encs_variational():
    from indl.model.beta_vae import make_encs_variational
    params = {
        'encs_rnn_units': 12,  # Only needed to emulate previous layer's outputs
        'dropout_rate': 0.01,
        'zs_size': 10,
        'qzs_off_diag': False,
        'qzs_init_std': 0.1,
        'q_samples': 2
    }
    encoded_s = tf.keras.Input(shape=(params['encs_rnn_units'],))
    qzs = make_encs_variational(params, encoded_s)
    # make_f_variational is a lightweight wrapper around `make_variational` which has its own tests.
    # There isn't much left to test here so let's just assert its type.
    assert K.is_keras_tensor(qzs)  # isinstance(qzs, tfd.MultivariateNormalDiag)

    make_encs_var = tf.keras.Model(inputs=encoded_s, outputs=qzs, name="make_encs_var_model")
    batch_size = 16
    encoded_s = tf.random.uniform((batch_size, params['encs_rnn_units']))
    dummy_qzs = make_encs_var(encoded_s)
    assert isinstance(dummy_qzs, tfd.MultivariateNormalDiag)
    dummy_sample = dummy_qzs.sample(params['q_samples'])
    assert dummy_sample.shape.as_list() == [params['q_samples'], batch_size, params['zs_size']]


def test_create_pzs():
    from indl.model.beta_vae import make_encs_variational, create_pzs

    params = {
        'encs_rnn_units': 12,  # Only needed to emulate encoding layer's outputs
        'dropout_rate': 0.01,
        'qzs_off_diag': False,

        'pzs_kappa': 0,
        'zs_size': 10,
        'qzs_init_std': 0.1,
        'q_samples': 2,
        'pzs_train_mean': True,
        'pzs_train_var': True,
        'pzs_off_diag': False
    }

    f_prior = create_pzs(params)
    assert isinstance(f_prior, tfd.MultivariateNormalDiag)

    # create_pzs is a lightweight wrapper around indl.dists.make_mvn_prior.
    # There are already tests for make_mvn_prior in test_tfp which tests all the combinations
    # of parameters. We don't need to test different parameters here.

    # To test the KL-divergence, we need to create an appropriate posterior distribution (qzs).
    # Let's use similar code to test_make_f_variational to generate qzs.
    encoded_s = tf.keras.Input(shape=(params['encs_rnn_units'],))
    qzs = make_encs_variational(params, encoded_s)
    make_var = tf.keras.Model(inputs=encoded_s, outputs=qzs, name="make_encs_var_model")
    batch_size = 16
    encoded_s = tf.random.uniform((batch_size, params['encs_rnn_units']))
    dummy_qzs = make_var(encoded_s)

    # Now calculate the KL-divergence between qzs and f_prior.
    # As qzs and f_prior have the same distribution type, the analytic
    # way to get KL divergence can be used, without drawing samples.
    f_kl = tfd.kl_divergence(dummy_qzs, f_prior)
    assert isinstance(f_kl, tf.Tensor)
    assert all(f_kl.numpy() > 0)


def test_create_encd(n_times=None):
    from indl.model.beta_vae import create_encd

    def default_params():
        return {
            'input_factors': 24,  # Pretend prepare_inputs used a read-in layer.
            'zs_size': 10,
            'q_samples': 1,

            'encd_rnn_type': 'BidirectionalGRUClip',
            'gru_clip_value': 5.0,
            'encd_rnn1_units': 16,
            'zd_lag': 1,
            'encd_rnn2_units': 14,
            'dec_rnn_type': None,  # Needs to be 'Complex' for LFADS, which defers rnn2 until later.
        }

    # Test LFADS formulation - note zs_sample=None
    K.clear_session()
    params = default_params()
    params['dec_rnn_type'] = 'Complex'
    inputs = tf.keras.Input(shape=(n_times, params['input_factors']))
    encoded_d = create_encd(params, inputs, zs_sample=None)
    assert encoded_d.shape[-1] == params['encd_rnn1_units']
    batch_size = 8
    timesteps = n_times or params['zd_lag'] + 10
    encoder_d = tf.keras.Model(inputs=inputs, outputs=encoded_d, name="encd_LFADS")
    dummy_inputs = tf.random.uniform((batch_size, timesteps, params['input_factors']))
    dummy_encoded_d = encoder_d(dummy_inputs)
    assert dummy_encoded_d.shape.as_list() == [batch_size, timesteps, params['encd_rnn1_units']]
    assert (dummy_encoded_d.numpy() != 0).sum() > 0

    # Test DSAE-Full formulation
    K.clear_session()
    params = default_params()
    params['q_samples'] = 1
    inputs = tf.keras.Input(shape=(n_times, params['input_factors']))
    zs_sample = tf.keras.Input(shape=(params['zs_size']))
    if params['q_samples'] > 1:
        # emulate a samples dimension.
        zs_sample = zs_sample[tf.newaxis, ...] + tf.zeros([params['q_samples'], 1, 1])
    encoded_d = create_encd(params, inputs, zs_sample=zs_sample)
    if params['q_samples'] > 1:
        assert len(encoded_d.shape.as_list()) == 4
        assert encoded_d.shape[0] == params['q_samples']
    else:
        assert len(encoded_d.shape.as_list()) == 3
    assert encoded_d.shape[-1] == params['encd_rnn2_units']
    # TODO: I don't think I can actually test a tf.keras.Model with one of its inputs having a "sample" dimension,
    #  so below test code only works with q_samples == 1. But > 1 should work in an end-to-end model.
    encoder_d = tf.keras.Model(inputs=(inputs, zs_sample), outputs=encoded_d, name="encd_DSAEFull")
    dummy_inputs = tf.random.uniform((batch_size, timesteps, params['input_factors']))
    dummy_zs_shape = (params['q_samples'],) if params['q_samples'] > 1 else ()
    dummy_zs_shape += (batch_size, params['zs_size'])
    dummy_zs_sample = tf.random.uniform(dummy_zs_shape)
    dummy_encoded_d = encoder_d((dummy_inputs, dummy_zs_sample))
    assert dummy_encoded_d.shape.as_list() == [batch_size, timesteps, params['encd_rnn2_units']]
    assert (dummy_encoded_d.numpy() != 0).sum() > 0

    # NODO: Test Full when zs_sample is still qzs. i.e., can we rely on implied .sample? No we can't.
    # _tmp = tf.keras.Input(shape=(params['zs_size']))
    # qzs = tfd.MultivariateNormalDiag(loc=_tmp)
    # encoded_d = create_encd(params, inputs, zs_sample=qzs)

    # Test DSAE-Factorized formulation. No RNN. Just a single z1 Dense layer.
    K.clear_session()
    params = default_params()
    params['encd_rnn_type'] = ''
    params['encd_rnn2_units'] = 0
    inputs = tf.keras.Input(shape=(n_times, params['input_factors']))
    encoded_d = create_encd(params, inputs, zs_sample=None)
    assert encoded_d.shape[-1] == params['encd_rnn1_units']
    encoder_d = tf.keras.Model(inputs=inputs, outputs=encoded_d, name="ecnd_DSAEFactorized")
    dummy_inputs = tf.random.uniform((batch_size, timesteps, params['input_factors']))
    dummy_encoded_d = encoder_d(dummy_inputs)
    assert dummy_encoded_d.shape.as_list() == [batch_size, timesteps, params['encd_rnn1_units']]
    assert (dummy_encoded_d.numpy() != 0).sum() > 0


def test_make_encd_variational(n_times=None):
    from indl.model.beta_vae import make_encd_variational

    params = {
        'encd_rnn2_units': 14,  # For constructing input
        'zd_size': 4,
        'qzd_init_std': 0.1,
        'qzd_off_diag': False,
        'q_samples': 1
    }
    inputs = tf.keras.Input(shape=(n_times, params['encd_rnn2_units']))
    qzd = make_encd_variational(params, inputs)
    assert K.is_keras_tensor(qzd)  # isinstance(qzd, tfd.MultivariateNormalDiag)

    z_var = tf.keras.Model(inputs=inputs, outputs=qzd)
    batch_size = 16
    timesteps = n_times or 20
    enc_z = tf.random.uniform((batch_size, timesteps, params['encd_rnn2_units']))
    dummy_qzd = z_var(enc_z)
    assert isinstance(dummy_qzd, tfd.MultivariateNormalDiag)
    dummy_sample = dummy_qzd.sample(params['q_samples'])
    assert dummy_sample.shape.as_list() == [params['q_samples'], batch_size, timesteps, params['zd_size']]


def _kl_test(dist, latent_dim, offdiag, init_std, timesteps, batch_size, encd_rnn2_units=16):
    # Test that we can do analytic KL.
    from indl.model.beta_vae import make_encd_variational
    params = {
        'zd_size': latent_dim,
        'qzd_off_diag': offdiag,
        'qzd_init_std': init_std,
        'q_samples': 1
    }
    inputs = tf.keras.Input(shape=(timesteps, encd_rnn2_units))
    qzd = make_encd_variational(params, inputs)
    # TODO: make_encd_variational should do tfd.Independent(_, reinterpreted_batch_ndims=1), then we don't do it below.
    make_var = tf.keras.Model(inputs=inputs, outputs=qzd)
    dummy_encd = tf.random.uniform((batch_size, timesteps, encd_rnn2_units))
    dummy_qzd = make_var(dummy_encd)
    dummy_qzd = tfd.Independent(dummy_qzd, reinterpreted_batch_ndims=1)
    kl_zd = tfd.kl_divergence(dummy_qzd, dist)
    assert isinstance(kl_zd, tf.Tensor)
    assert np.all(kl_zd.numpy() > 0)


def test_create_pzd():
    from indl.model.beta_vae import create_pzd
    params = {
        'pzd_process': 'RNN',  # 'AR1' or an RNN cell type or 'none' for Gaussian
        'zd_size': 4,
        'pzd_units': 3,
        'pzd_tau': 10.0,
        'pzd_init_std': 0.1,
        'pzd_train_mean': True,
        'pzd_train_tau': True,
        'pzd_train_var': True,
        'pzd_offdiag': False,
    }
    zd_generator = create_pzd(params)

    timesteps = 20
    batch_size = 16
    sample, dist = zd_generator.get_dist(timesteps, samples=1, batch_size=batch_size)
    _kl_test(dist, params['zd_size'], params['pzd_offdiag'], params['pzd_init_std'],
             timesteps, batch_size, encd_rnn2_units=16)


def test_create_decoder():
    from indl.model.beta_vae import create_decoder

    q_samples = 2  # For constructing zs_sample and z_sample
    zs_size = 10  # For constructing zs_sample
    n_times = 20  # Number of time steps in enc_z
    zd_size = 4  # For constructing enc_z

    # params for create_generator
    params = {
        'dec_rnn_type': 'GRU',  # Can be any of the supported RNN cell types
        'zs_to_dec': 'init',  # 'initial state' or 'tiled to input' - determines how f is used in generator.
        'gru_clip_value': 5.0,  # Only for GRU or BidirectionalGRU
        'dec_rnn_units': 8,
        'n_factors': 10,
        'dropout_rate': 1e-2,  # After generator output, before to-factors Dense layer.
    }

    # Create placeholders for f_enc and z_enc
    encoded_s = tf.keras.Input(shape=(zs_size,))
    # Reproduce the correct tensor shape for going through make_encs_variational and sampling.
    zs_sample = tf.expand_dims(encoded_s, 0) + tf.zeros((q_samples, 1, 1))

    # Normal --> enc_z is sample from q(z_t)
    encoded_d = tf.keras.Input(shape=(n_times, zd_size))
    # Pretend it went through make_variational and .sample()
    zd_sample = tf.expand_dims(encoded_d, 0) + tf.zeros((q_samples, 1, 1, 1))

    gen_outputs, factors = create_decoder(params, zs_sample, zd_sample)
    assert gen_outputs.shape.as_list() == [q_samples, None, n_times, params['dec_rnn_units']]
    assert factors.shape.as_list() == [q_samples, None, n_times, params['n_factors']]

    # Full test with dummys - still can't seem to do this with a 'sample' dimension on tf.keras.Model
    _zs = tf.keras.Input(shape=(None, zs_size))
    _zd = tf.keras.Input(shape=(None, n_times, zd_size))
    _gen_state, _facs = create_decoder(params, _zs, _zd)
    gen_model = tf.keras.Model(inputs=(_zs, _zd), outputs=(_gen_state, _facs))

    batch_size = 16
    dummy_zs = tf.random.uniform((q_samples, batch_size, zs_size))
    dummy_zd = tf.random.uniform((q_samples, batch_size, n_times, zd_size))
    dummy_gen, dummy_facs = gen_model((dummy_zs, dummy_zd))
    assert dummy_gen.shape.as_list() == [q_samples, batch_size, n_times, params['dec_rnn_units']]
    assert not np.any(np.isnan(dummy_gen.numpy()))
    assert dummy_facs.shape.as_list() == [q_samples, batch_size, n_times, params['n_factors']]
    assert not np.any(np.isnan(dummy_facs.numpy()))


def test_create_decoder_complex():
    from indl.model.beta_vae import create_decoder_complex

    zs_size = 10  # For constructing zs_sample
    q_zs_samples = 1  # For constructing zs_sample. Couldn't figure out how to get it to work with LFADS
    n_times = 20  # Number of time steps in z1
    encd_rnn1_units = 64  # For constructing z1 placeholder/dummy

    # params for create_generator
    params = {
        'dec_rnn_type': 'Complex',  # Must be "Complex" for LFADS
        'encd_rnn2_units': 30,      # Number of units in z2 RNN in LFADS ComplexCell; LFADS' con_hidden_state_dim
        'zd_size': 12,              # Z latent size: z2 states -> Dense(zd_size)
        'gru_clip_value': 5.0,      # Only for GRU or BidirectionalGRU
        'dec_rnn_units': 100,
        'n_factors': 10,
        'dropout_rate': 1e-2,       # After generator output, before to-factors Dense layer.
        'ext_input_dim': 0,
    }

    # Create placeholders for encoded_zs and encoded_zd1
    encoded_zs = tf.keras.Input(shape=(zs_size,))
    # Reproduce the correct tensor shape for going through make_f_variational and sampling.
    if q_zs_samples > 1:
        encoded_zs = tf.expand_dims(encoded_zs, 0) + tf.zeros((q_zs_samples, 1, 1))
        encoded_zs = tfkl.Lambda(lambda x: x)(encoded_zs)
    encoded_zd1 = tf.keras.Input(shape=(n_times, encd_rnn1_units))
    ext_input = tf.keras.Input(shape=(n_times, params['ext_input_dim']), name="ext_input")

    result = create_decoder_complex(params, encoded_zs, encoded_zd1, ext_input=ext_input)
    gen_outputs, factors, enczd2_state, qzd_mean, qzd_logvar, qzd_sample = result
    assert gen_outputs.shape.as_list() == [None, n_times, params['dec_rnn_units']]
    assert factors.shape.as_list() == [None, n_times, params['n_factors']]

    # TODO: Dummy test


def test_output_dist():
    from indl.model.beta_vae import output_dist
    n_factors = 10
    n_times = 20

    params = {
        'output_dist': 'Poisson',
        'output_dist_offdiag': False,
        'out_dim': 32
    }

    factors = tfkl.Input(shape=(None, n_factors))
    q_rates = output_dist(factors, params)

    model = tf.keras.Model(inputs=factors, outputs=q_rates)
    batch_size = 16
    dummy_factors = tf.random.uniform((batch_size, n_times, n_factors))
    result = model(dummy_factors)
    out_dist = tfd.Independent(result, reinterpreted_batch_ndims=2)
    assert out_dist.event_shape.as_list() == [n_times, params['out_dim']]
    assert out_dist.batch_shape.as_list() == [batch_size]

    # TODO: KL test

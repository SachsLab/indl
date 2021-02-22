__all__ = ['generate_default_args', 'generate_default_params', 'prepare_inputs',
           'create_f_encoder', 'make_f_variational', 'create_z_encoder', 'make_z_variational',
           'get_z_prior']


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from .lfads.utils import CoordinatedDropout
from .lfads.utils import GRUClip
# from indl.model.tfp import LearnableMultivariateNormalDiag
from .tfp import make_mvn_prior, make_mvn_dist_fn
from .tfp import LearnableMultivariateNormalDiagCell
from .lfads.utils import LearnableAutoRegressive1Prior

tfd = tfp.distributions
tfpl = tfp.layers


def generate_default_args():
    # non tunable parameters
    return type('TestArgs', (object,), dict(
        random_seed=1337,
        batch_size=16,
        n_epochs=100,
        resample_X=10,                   # spike count bin size
        f_rnn_type="BidirectionalGRUClip",  # Encoder RNN cell type: ('Bidirectional' or '') +  ('GRU', 'LSTM', 'SimpleRNN', 'GRUClip')
        f_latent_off_diag=False,         # If latent dist may have non-zero off diagonals
        q_f_init_std=0.1,                # ic_prior_var
        q_f_samples=1,
        f_prior_off_diag=False,          # If latent prior may have non-zero off diagonals
        f_prior_kappa=0.0,               # In LFADS this is a tuned hyperparameter, ~0.1
        f_prior_train_mean=True,         # True in LFADS
        f_prior_train_var=True,          # False in LFADS
        f_enc_input_samps=0,             # Set to > 0 to restrict f_enc to only see this many samples, to prevent acausal
        z1_rnn_type="BidirectionalGRUClip",  # Encoder RNN cell type
        z1_lag=0,                        # Time lag on the z-encoder output.
                                         #  Same as LFADS' `controller_input_lag`
        q_z_init_std=0.1,                #
        z_prior_var=0.1,                 # co_prior_var
        z_prior_process="RNN",           # RNN or AR1
        latent_samples=4,                # Currently unused
        gen_cell_type="GRUClip",         # Decoder generative RNN cell type. "Complex" is for LFADS.
        gen_tile_input=False,
    ))


def generate_default_params():
    # tunable parameters
    return {
        "dropout_rate": 1e-2,               # (1e-2)
        "coordinated_dropout_rate": 0.1,    #
        "input_factors": 0,                 # Extra Dense layer applied to inputs. Good for multi-session.
        "gru_clip_value": 5.0,              # Max value recurrent cell can take before being clipped (5.0)
        "f_units": [128],                   # Number of units in f-encoder RNN. Increase list length to add more RNN layers. (128)
                                            #  Same as LFADS' `ic_enc_dim`
        "f_latent_size": 10,                # Size of latent vector f (10)
                                            #  Same as LFADS' `ic_dim`
        "z1_units": 16,                     # Number of units in z-encoder RNN.
                                            #  Same as LFADS `ci_enc_dim`
        "z2_units": 16,                     # Number of units in z2 RNN, in DHSAE Full or LFADS controller.
                                            #  Same as LFADS `con_dim`
        "z_latent_size": 4,                 # Dimensionality of q_zt posterior.
                                            #  Same as LFADS' `co_dim`
        "gen_n_hidden": 256,                # Number of RNN cells in generator (256)
                                            #  Same as LFADS `gen_dim`
        "n_factors": 10,                    # Number of latent factors (24)
                                            #  Same as LFADS' `factors_dim`
        "gen_l2_reg": 1e-4,                 # (1e-4)
        "learning_rate": 2e-3,              # (2e-3)


        # "max_grad_norm": 200.0
    }
# Anecdotally
# -larger f_units (~128) is important to get latents that discriminate task
# -larger gen_n_hidden (~256) is important to get good reconstruction


_args = generate_default_args()
_params = generate_default_params()


def prepare_inputs(params, _inputs):
    _inputs = tfkl.Dropout(params['dropout_rate'])(_inputs)

    # The f-encoder takes the entire sequence and outputs a single-timestamp vector,
    # this vector is used as the decoder's initial condition. This has the potential
    # to create acausal modeling because the decoder will have knowledge of the entire
    # sequence from its first timestep.
    # We can optionally split the input to _f_enc_inputs and remaining _inputs
    # RNN will only see _f_enc_inputs to help prevent acausal modeling.
    _f_enc_inputs = _inputs[:, :params['f_enc_input_samps'], :]
    _inputs = _inputs[:, params['f_enc_input_samps']:, :]

    # Coordinated dropout on _inputs only.
    # Why not _f_enc_inputs? Is it because it is likely too short to matter?
    _masked_inputs, cd_kept_mask = CoordinatedDropout(params['coordinated_dropout_rate'])(_inputs)
    # cd_kept_mask is part of the return so it can be used during decoding.

    # The z-encoder inputs will always be full length.
    _z_enc_inputs = tf.concat([_f_enc_inputs, _masked_inputs], axis=-2)

    if params['f_enc_input_samps'] == 0:
        # With no f_enc_input_samps specification, the f_enc inputs are the full input.
        #  Note this has coordinated dropout, whereas it wouldn't if f_enc_input_samps was specified.
        _f_enc_inputs = _masked_inputs

    # Note: Skipping over CV Mask

    if params['input_factors'] > 0:
        _f_enc_inputs = tfkl.Dense(params['input_factors'])(_f_enc_inputs)
        _z_enc_inputs = tfkl.Dense(params['input_factors'])(_z_enc_inputs)

    return _f_enc_inputs, _z_enc_inputs, cd_kept_mask


def test_prepare_inputs(n_times=None, n_sensors=36):
    K.clear_session()
    inputs = tf.keras.Input(shape=(n_times, n_sensors))
    f_enc_inputs, z_enc_inputs, cd_mask = prepare_inputs(
        {**_params, **_args.__dict__}, inputs)
    test_model = tf.keras.Model(inputs=inputs, outputs=[f_enc_inputs, z_enc_inputs, cd_mask])
    test_model.summary()

    test_output = test_model(tf.random.uniform((_args.batch_size, n_times or 2, n_sensors)))
    print([(_.shape, _.dtype) for _ in test_output])


def create_f_encoder(params, _inputs,
                     kernel_initializer='lecun_normal',
                     bias_initializer='zeros',
                     recurrent_regularizer='l2'):
    """
    The f-encoder.
    Also called "Static Encoder", or in LFADS the "initial condition encoder".
    """
    _latents = _inputs

    rnn_kwargs = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        recurrent_regularizer=recurrent_regularizer,
        dropout=0,  # Dropout on inputs not needed.
        return_sequences=False)
    if params['f_rnn_type'].endswith('GRU'):
        rnn_layer_cls = tfkl.GRU
    elif params['f_rnn_type'].endswith('LSTM'):
        rnn_layer_cls = tfkl.LSTM
    elif params['f_rnn_type'].endswith('SimpleRNN'):
        rnn_layer_cls = tfkl.SimpleRNN
    elif params['f_rnn_type'].endswith('GRUClip'):
        rnn_layer_cls = GRUClip
        rnn_kwargs['clip_value'] = params['gru_clip_value']

    for ix, rnn_units in enumerate(params['f_units']):
        if params['f_rnn_type'].startswith('Bidirectional'):
            _latents = tfkl.Bidirectional(rnn_layer_cls(rnn_units, **rnn_kwargs),
                                          merge_mode="concat",
                                          name="f_rnn_" + str(ix))(_latents)
        else:
            _latents = rnn_layer_cls(rnn_units, **rnn_kwargs)(_latents)

    return _latents


def make_f_variational(params, _enc_f):
    _enc_f = tfkl.Dropout(params['dropout_rate'])(_enc_f)

    # Use a helper function to get a MVN distribution from _latents.
    make_dist_fn, dist_params = make_mvn_dist_fn(_enc_f, params['f_latent_size'],
                                                 shift_std=params['q_f_init_std'],
                                                 offdiag=params['f_latent_off_diag'],
                                                 loc_name="f_loc", scale_name="f_scale",
                                                 use_mvn_diag=True)
    _q_f = tfpl.DistributionLambda(make_distribution_fn=make_dist_fn,
                                   # convert_to_tensor_fn=lambda s: s.sample(N_SAMPLES),
                                   name="q_f")(dist_params)

    # Also return a matching prior. This will be used in test_step to measure KL.
    if params['f_prior_kappa'] > 0:
        raise NotImplementedError
    prior = make_mvn_prior(params['f_latent_size'],
                           init_std=params['q_f_init_std'],
                           trainable_mean=params['f_prior_train_mean'],
                           trainable_var=params['f_prior_train_var'],
                           offdiag=params['f_prior_off_diag'])
    # prior_factory = lambda: tfd.MultivariateNormalDiag(loc=0, scale_diag=params['f_prior_kappa'])
    # prior_factory = LearnableMultivariateNormalDiag(params['f_latent_size'])
    # prior_factory.build(input_shape=(0,))

    return _q_f, prior


def test_create_f_encoder(n_times=None, n_sensors=36):
    K.clear_session()

    input_samps = _args.f_enc_input_samps or n_times
    input_dim = _params['input_factors'] or n_sensors
    f_enc_inputs = tf.keras.Input(shape=(input_samps, input_dim))
    latents = create_f_encoder({**_params, **_args.__dict__}, f_enc_inputs)
    q_f, f_prior = make_f_variational({**_params, **_args.__dict__}, latents)

    f_encoder = tf.keras.Model(inputs=f_enc_inputs, outputs=q_f, name="f_encoder_model")
    f_encoder.summary()

    dummy_q_f = f_encoder(tf.random.uniform((_args.batch_size, input_samps or 2, input_dim)))
    print("q_f: ", dummy_q_f)
    print("q_f.sample(2).shape (samples, batch_size, f_dim): ", dummy_q_f.sample(2).shape)

    # f_prior = f_prior_factory()
    f_kl = tfd.kl_divergence(dummy_q_f, f_prior)
    print("f KL: ", f_kl)


def create_z_encoder(params, _inputs,
                     _f_inputs=None,
                     f_inputs_pre_z1=True,
                     kernel_initializer='lecun_normal',
                     bias_initializer='zeros',
                     recurrent_regularizer='l2'):
    # For LFADS, set _f_inputs=None and params['gen_cell_type']="Complex"

    if _f_inputs is not None:
        # Expand along time dimension by broadcast-add to zeros.
        n_times = tf.shape(_inputs)[-2]
        exp_zeros = tf.zeros(tf.stack((n_times, 1)))
        _f_inputs = tf.expand_dims(_f_inputs, -2) + exp_zeros

    # Add optional f_input that we tile and concatenate onto _inputs.
    if _f_inputs is not None and f_inputs_pre_z1:
        if params['q_f_samples'] > 1:
            # _inputs needs to be repeated NUM_SAMPLES on a new samples axis at axis=0.
            _inputs = _inputs[tf.newaxis, ...] + tf.zeros([params['q_f_samples'], 1, 1, 1])
        # Concatenate _x2 (features) and _static_sample
        _inputs = tfkl.Concatenate()([_inputs, _f_inputs])  # (optional-samples, batch, timesteps, feat_dim+latent_static)
        # Collapse samples + batch dims  -- required by LSTM
        new_d1 = tf.reshape(tf.reduce_prod(tf.shape(_inputs)[:-2]), (1,))
        new_shape = tf.concat((new_d1, tf.shape(_inputs)[-2:]), 0)
        _inputs = tf.reshape(_inputs, new_shape)
        # _inputs shape now (samples*batch, T, feat+lat_stat)

    is_rnn = params['z1_rnn_type'].startswith('Bidirectional') \
        or (params['z1_rnn_type'] in ['GRU', 'LSTM', 'SimpleRNN', 'GRUClip'])
    if is_rnn:
        rnn_kwargs = dict(
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            recurrent_regularizer=recurrent_regularizer,
            dropout=0,  # Dropout on inputs not needed.
            return_sequences=True)
        if params['z1_rnn_type'].endswith('GRU'):
            rnn_layer_cls = tfkl.GRU
        elif params['z1_rnn_type'].endswith('LSTM'):
            rnn_layer_cls = tfkl.LSTM
        elif params['z1_rnn_type'].endswith('SimpleRNN'):
            rnn_layer_cls = tfkl.SimpleRNN
        elif params['z1_rnn_type'].endswith('GRUClip'):
            rnn_layer_cls = GRUClip
            rnn_kwargs['clip_value'] = params['gru_clip_value']

        if params['z1_rnn_type'].startswith('Bidirectional'):
            _enc_z = tfkl.Bidirectional(rnn_layer_cls(params['z1_units'], **rnn_kwargs),
                                        merge_mode="concat", name="z_rnn_1")(_inputs)
        else:
            _enc_z = rnn_layer_cls(params['z1_units'], **rnn_kwargs)(_inputs)
    else:
        # Not RNN, just MLP
        _enc_z = tfkl.Dense(params['z1_units'])(_inputs)

    if params['z1_lag'] > 0:
        raise NotImplementedError
        # TODO: Split output to forward and backward,
        #  For forward, trim off ending `toffset` and prepend zeros
        #  if rnn and bidirectional, for backward, trim off beginning `toffset` and append zeros

    if is_rnn and params['z1_rnn_type'].startswith('Bidirectional'):
        # TODO: Recombine forward and backward with equivalent to merge_mode="sum"
        pass

    if params['z2_units'] > 0 and params['gen_cell_type'] != "Complex":
        if _f_inputs is not None and not f_inputs_pre_z1:
            if params['q_f_samples'] > 1:
                # _inputs needs to be repeated NUM_SAMPLES on a new samples axis at axis=0.
                _enc_z = _enc_z[tf.newaxis, ...] + tf.zeros([params['q_f_samples'], 1, 1, 1])
            # Concatenate _x2 (features) and _static_sample
            _enc_z = tfkl.Concatenate()(
                [_enc_z, _f_inputs])  # (optional-samples, batch, timesteps, feat_dim+latent_static)
            # Collapse samples + batch dims  -- required by LSTM
            new_d1 = tf.reshape(tf.reduce_prod(tf.shape(_enc_z)[:-2]), (1,))
            new_shape = tf.concat((new_d1, tf.shape(_enc_z)[-2:]), 0)
            _enc_z = tf.reshape(_enc_z, new_shape)
            # _enc_z shape now (samples*batch, T, feat+lat_stat)

        # z2 vanilla RNN used in DHSAE Full. LFADS' z2 used elsewhere.
        _ = rnn_kwargs.pop('clip_value', None)
        _enc_z = tfkl.SimpleRNN(params['z2_units'], **rnn_kwargs)(_enc_z)

    return _enc_z


def make_z_variational(params, _enc_z):
    """
    Take the encoded latent sequence z (output of z1 and optionally z2)
    and convert it to a distribution.

    This isn't necessary for LFADS models because z isn't in its final encoded
    form until inside the Complex cell, so it's up to the complex cell to
    handle the formation of the distribution.
    """
    if params['gen_cell_type'] == "Complex":
        # LFADS - variational part taken care of in complex cell.
        return _enc_z

    # Get a multivariate normal diag over each timestep.
    make_dist_fn, dist_params = make_mvn_dist_fn(
        _enc_z, params['z_latent_size'], shift_std=params['q_z_init_std'],
        offdiag=False, loc_name="z_loc", scale_name="z_scale", use_mvn_diag=True)
    _q_z = tfpl.DistributionLambda(make_distribution_fn=make_dist_fn,
                                   # convert_to_tensor_fn=lambda s: s.sample(N_SAMPLES),
                                   name="q_z")(dist_params)
    return _q_z


def get_z_prior(params):

    # TODO: Also return appropriate prior
    if params['z_prior_process'] == 'AR1':
        prior = LearnableAutoRegressive1Prior(graph_batch_size, hps.co_dim,
                                          autocorrelation_taus,
                                          noise_variances,
                                          hps.do_train_prior_ar_atau,
                                          hps.do_train_prior_ar_nvar,
                                          "u_prior_ar1")
    else:
        # RNN
        prior = LearnableMultivariateNormalDiagCell(self.hidden_size, self.latent_size,
                                                                 cell_type='gru')

    return prior


def test_create_z_encoder(n_times=None, n_sensors=36):
    # by skipping over prepare_inputs we are ignoring any read-in layers and coordinated dropout
    K.clear_session()

    input_dim = _params['input_factors'] or n_sensors
    inputs = tf.keras.Input(shape=(n_times, input_dim))
    f_sample = tf.keras.Input(shape=(_params['f_latent_size']))
    z_enc = create_z_encoder({**_params, **_args.__dict__}, inputs, _f_inputs=f_sample)
    q_z = make_z_variational({**_params, **_args.__dict__}, z_enc)

    z_encoder = tf.keras.Model(inputs=(inputs, f_sample), outputs=q_z, name="z_encoder_model")
    z_encoder.summary()

    dummy_q_z = z_encoder(tf.random.uniform((_args.batch_size, n_times or 2, input_dim)),
                          tf.random.uniform((_args.batch_size, _params['f_latent_size'])))
    print(type(dummy_q_z), dummy_q_z.shape)

    # TODO: Test KL divergence
    z_prior = get_z_prior({**_params, **_args.__dict__})
    z_kl = tfd.kl_divergence(dummy_q_z, z_prior)
    print("z KL: ", z_kl)


def main():
    # test_prepare_inputs()
    # test_create_f_encoder()
    test_create_z_encoder()


if __name__ == "__main__":
    main()

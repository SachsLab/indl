from typing import Tuple, Union, Optional
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
from indl.layers import CoordinatedDropout
from indl.rnn.gru_clip import GRUClip
from indl.dists import make_mvn_prior
from indl.dists.sequential import IProcessMVNGenerator


tfd = tfp.distributions
tfpl = tfp.layers


def generate_default_args():
    """

    Returns: non-tunable parameters in a TestArgs object
    Access args with obj.arg_name or .__dict__['arg_name']

    """
    # non tunable parameters
    return type('TestArgs', (object,), dict(
        random_seed=1337,
        batch_size=16,
        n_epochs=100,
        resample_X=10,                          # spike count bin size

        q_samples=1,                            # At some points this gets folded into the batch dim so it has to
                                                #  be the same for q_f and q_z.
        # Encoder - Static
        encs_rnn_type="BidirectionalGRUClip",   # Encoder RNN cell type: ('Bidirectional' or '')
                                                # + ('GRU', 'LSTM', 'SimpleRNN', 'GRUClip')
        encs_input_samps=0,                     # Set to > 0 to restrict f_encoder to only see this many samples
                                                #  This is one of several settings required to prevent acausal modeling.
        qzs_off_diag=False,                     # If latent dist may have non-zero off diagonals
        # Static Latent Dist
        qzs_init_std=0.1,                       # (LFADS: ic_prior_var)
        # Static Latent Prior
        pzs_off_diag=False,                     # If latent prior may have non-zero off diagonals
        pzs_kappa=0.0,                          # In LFADS this is a tuned hyperparameter, ~0.1
        pzs_train_mean=True,                    # True in LFADS
        pzs_train_var=True,                     # False in LFADS

        # Encoder - Dynamic
        encd_rnn_type="BidirectionalGRUClip",   # Encoder RNN cell type
                                                #  To prevent acausal modeling on the controller input, do not use Bidir
        zd_lag=0,                               # Time lag on the z-encoder output.
                                                #  Same as LFADS' `controller_input_lag`
        # Dynamic Latent Dist
        qzd_init_std=0.1,                       # std shift when z-latent is 0,
                                                # and initial prior variance for RNN and tiled gaussian priors
        qzd_off_diag=False,
        # Dynamic Latent Prior
        pzd_process="AR1",                      # AR1 or a RNN cell type, or anything else 'none' for
                                                # simple tiled gaussian.
        pzd_train_mean=False,                   #
        pzd_train_var=True,                     # Also used for train_nvar
        pzd_init_std=0.1,                       # Also used for inittau
        pzd_offdiag=False,                      #
        pzd_units=8,                            # Number of units for RNN MVN prior (RNNMVNGenerator)
        pzd_tau=10.0,                           # Initial autocorrelation for AR(1) priors (AR1ProcessMVNGenerator)
        pzd_train_tau=True,                     #

        # Decoder
        dec_rnn_type="GRUClip",                 # Decoder generative RNN cell type. "Complex" is for LFADS.
        zs_to_dec="initial conditions",         # How static latent is used in the decoder.
                                                #  "initial conditions" or "tile inputs"

        # Output
        output_dist="Poisson",                  # Poisson or anything else for MVNDiag
    ))


def generate_default_params() -> dict:
    """

    Returns: tunable parameters in dictionary

    """
    # tunable parameters
    return {
        "dropout_rate": 1e-2,               # (1e-2)
        "coordinated_dropout_rate": 0.1,    #
        "input_factors": 0,                 # Extra Dense layer applied to inputs. Good for multi-session. (not impl.)
        "gru_clip_value": 5.0,              # Max value recurrent cell can take before being clipped (5.0)
        "gen_l2_reg": 1e-4,  # (1e-4)
        "learning_rate": 2e-3,  # (2e-3)
        # "max_grad_norm": 200.0

        # Encoder - Static
        "encs_rnn_units": [128],            # Number of units in static encoder RNN.
                                            #  Increase list length to add more RNN layers. (128)
                                            #  Same as LFADS' `ic_enc_dim`
        "zs_size": 10,                      # Size of static latent vector zs (10)
                                            #  Same as LFADS' `ic_dim`

        # Encoder - Dynamic
        "encd_rnn1_units": 16,              # Number of units in dynamic encoder first RNN.
                                            #  Same as LFADS `ci_enc_dim`
        "encd_rnn2_units": 16,              # Number of units in dynamic encoder second RNN (DHSAE Full or LFADS con).
                                            #  Same as LFADS `con_dim`
        "zd_size": 4,                       # Dimensionality of q_zt posterior.
                                            #  Same as LFADS' `co_dim`

        # Decoder
        "dec_rnn_units": 256,               # Number of RNN cells in decoder RNN (256)
                                            #  Same as LFADS `gen_dim`
        "n_factors": 10,                    # Number of latent factors (24)
                                            #  Same as LFADS' `factors_dim`
    }
# Anecdotally
# -larger encs_rnn_units (~128) is important to get latents that discriminate task
# -larger dec_rnn_units (~256) is important to get good reconstruction


_args = generate_default_args()
_params = generate_default_params()


def prepare_inputs(params: dict, _inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Prepare the data for entry into the encoder(s).
    This comprises several steps:

    * dropout
    * (optional) split off inputs to the f_encoder to prevent acausal modeling
    * (optional) coordinated dropout
    * (not implemented) CV mask
    * (optional) Dense layer to read-in inputs to a common set of input factors.

    To keep the model flexible to inputs of varying timesteps, this model fragment does
    not check the size of the timestep dimension. Please make sure that params['encs_input_samps'] is
    less than the smallest number of timesteps in your inputs.

    Args:
        params: has the following keys
            - 'dropout_rate'
            - 'encs_input_samps' - set to > 0 to split off f_encoder inputs to prevent acausal modeling.
            - 'coordinated_dropout_rate'
            - 'input_factors'
        _inputs: With dimensions (batch, timesteps, features)

    Returns:
        f_enc_inputs: to be used as inputs to a subsequent f_encoder.
            If params['encs_input_samps'] > 0 then this will simply be the leading slice off inputs unmasked,
            else this will be full length and masked. (In both cases it will be optionally run through Dense layer).
        z_enc_inputs: to be used as inputs to a subsequent z_encoder
        cd_kept_mask: with dtype tf.bool, to be used during decoding for "coordinated dropout"
    """
    _inputs = tfkl.Dropout(params['dropout_rate'])(_inputs)

    # The f-encoder takes the entire sequence and outputs a single-timestamp vector,
    # this vector is used as the decoder's initial condition. This has the potential
    # to create acausal modeling because the decoder will have knowledge of the entire
    # sequence from its first timestep.
    # We can optionally split the input to _f_enc_inputs and remaining _inputs
    # RNN will only see _f_enc_inputs to help prevent acausal modeling.
    _f_enc_inputs = _inputs[:, :params['encs_input_samps'], :]
    _inputs = _inputs[:, params['encs_input_samps']:, :]

    # Coordinated dropout on _inputs only.
    # Why not _f_enc_inputs? Is it because it is likely too short to matter?
    _masked_inputs, cd_kept_mask = CoordinatedDropout(params['coordinated_dropout_rate'])(_inputs)
    # cd_kept_mask is part of the return so it can be used during decoding.

    # The z-encoder inputs will always be full length.
    _z_enc_inputs = tf.concat([_f_enc_inputs, _masked_inputs], axis=-2)

    if params['encs_input_samps'] == 0:
        # With no encs_input_samps specification, the f_enc inputs are the full input.
        #  Note this has coordinated dropout, whereas it wouldn't if encs_input_samps was specified.
        _f_enc_inputs = _masked_inputs

    # Note: Skipping over LFADS' CV Mask for now.

    if params['input_factors'] > 0:
        _f_enc_inputs = tfkl.Dense(params['input_factors'])(_f_enc_inputs)
        _z_enc_inputs = tfkl.Dense(params['input_factors'])(_z_enc_inputs)

    return _f_enc_inputs, _z_enc_inputs, cd_kept_mask


def create_encs(params: dict, inputs: tf.Tensor,
                kernel_initializer='lecun_normal',
                bias_initializer='zeros',
                recurrent_regularizer='l2') -> tf.Tensor:
    """
    The static arm of the encoder, aka in LFADS as the "initial condition encoder".

    Args:
        params: required keys are 'encs_rnn_units' (int or iterable of ints), 'encs_rnn_type' (str), 'gru_clip_value'
        inputs: a tensor with dimensions (batch_size, timesteps, input_dim)
            batch_size and timesteps may be `None` for placeholder tensors (i.e., created by tf.keras.Input)
        kernel_initializer: see TF's RNN docs
        bias_initializer: see TF's RNN docs
        recurrent_regularizer: see TF's RNN docs

    Returns:
        statically encoded x (not variational)

    """

    _encoded_s = inputs

    encs_rnn_units = params['encs_rnn_units']
    if not isinstance(encs_rnn_units, (list, tuple)):
        encs_rnn_units = [encs_rnn_units]

    rnn_kwargs = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        recurrent_regularizer=recurrent_regularizer,
        dropout=0,  # Dropout on inputs not needed.
        )
    if params['encs_rnn_type'].endswith('GRU'):
        rnn_layer_cls = tfkl.GRU
    elif params['encs_rnn_type'].endswith('LSTM'):
        rnn_layer_cls = tfkl.LSTM
    elif params['encs_rnn_type'].endswith('SimpleRNN'):
        rnn_layer_cls = tfkl.SimpleRNN
    elif params['encs_rnn_type'].endswith('GRUClip'):
        rnn_layer_cls = GRUClip
        rnn_kwargs['clip_value'] = params['gru_clip_value']

    for ix, rnn_units in enumerate(encs_rnn_units):
        rnn_kwargs['return_sequences'] = (ix + 1) < len(encs_rnn_units)
        if params['encs_rnn_type'].startswith('Bidirectional'):
            _encoded_s = tfkl.Bidirectional(rnn_layer_cls(rnn_units, **rnn_kwargs),
                                            merge_mode="concat",
                                            name="rnn_s_" + str(ix))(_encoded_s)
        else:
            _encoded_s = rnn_layer_cls(rnn_units, **rnn_kwargs)(_encoded_s)

    return _encoded_s


def make_encs_variational(params: dict, encoded_s: tf.Tensor) \
        -> Union[tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL]:
    """
    Make the output of the Static Encoder (encs) variational.
    Adds a dropout layer and passes through indl.model.tfp.make_variational with the correct parameters.
    Args:
        params:
            - 'dropout_rate'
            - 'zs_size'
            - 'qzs_off_diag'
            - 'qzs_init_std'
            - 'q_samples'
        encoded_s:

    Returns:
        q(zs|x)
    """
    from indl.dists import make_variational

    encoded_s = tfkl.Dropout(params['dropout_rate'])(encoded_s)

    qzs = make_variational(encoded_s, params['zs_size'],
                           init_std=params['qzs_init_std'],
                           offdiag=params['qzs_off_diag'],
                           samps=params['q_samples'],
                           loc_name="f_loc", scale_name="f_scale",
                           use_mvn_diag=True)
    return qzs


def create_pzs(params: dict) -> Union[tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL]:
    """
    Make a prior with optionally trainable mean and variance.
    Args:
        params:
            - 'pzs_kappa' -- must be 0.
            - 'zs_size'
            - 'qzs_init_std'
            - 'pzs_train_mean'
            - 'pzs_train_var'
            - 'pzs_off_diag'
    Returns:
        Either tfd.MultivariateNormalTril if params['pzs_off_diag'] else tfd.MultivariateNormalDiag
    """
    if params['pzs_kappa'] > 0:
        raise NotImplementedError
    pzs = make_mvn_prior(params['zs_size'],
                         init_std=params['qzs_init_std'],
                         trainable_mean=params['pzs_train_mean'],
                         trainable_var=params['pzs_train_var'],
                         offdiag=params['pzs_off_diag'])
    # Old way:
    # prior_factory = lambda: tfd.MultivariateNormalDiag(loc=0, scale_diag=params['pzs_kappa'])
    # prior_factory = LearnableMultivariateNormalDiag(params['zs_size'])
    # prior_factory.build(input_shape=(0,))
    return pzs


def create_encd(params: dict, inputs: tf.Tensor,
                zs_sample: Optional[int] = None,
                f_inputs_pre_z1: bool = True,
                kernel_initializer: str = 'lecun_normal',
                bias_initializer: str = 'zeros',
                recurrent_regularizer: str = 'l2') -> tf.Tensor:
    """
    Run the input through the Dynamic Encoder (aka LFADS' controller input encoder).
    Different formulations in the literature:
    DSAE static: z not used
    DSAE dynamic full:
        - params['encd_rnn_type'] indicates Bidirectional RNN of some cell type
        - params['encd_rnn2_units'] > 0
        - zs_sample is a tensor, possibly with a leading 'samples' dimension.
    DSAE dynamic factorized:
        - params['encd_rnn_type'] can be None or something nonsense.
        - params['encd_rnn2_units'] = 0
        - zs_sample = None
    LFADS (simple z1 only because f1-joining and z2 encoding happens in its ComplexCell):
        - params['encd_rnn_type'] = 'BidirectionalGRU' (any will do)
        - params['dec_rnn_type'] != 'Complex'
        - params['encd_rnn2_units'] = 0  # TODO: I want to reuse encd_rnn2_units to parameterize LFADS' complex cell's internal GRU.
        - zs_sample = None
    Args:
        params:
            - 'encd_rnn_type': Type of RNN. '' or 'Bidirectional' + one of ['GRU', 'LSTM', 'SimpleRNN', 'GRUClip']
                (just like create_f_encoder's 'encs_rnn_type' param), OR something else to not use an RNN and just use
                a flat Dense layer. Do not use 'Bidirectional' prefix for causal modeling.
            - 'gru_clip_value': Required if encd_rnn_type endswith GRUClip
            - 'encd_rnn1_units': Number of units in the first-level z encoder layer.
            - 'zd_lag': simulate a delay in the z1 outputs.
            - 'encd_rnn2_units': Number of units in the second-level z encoder layer. Can be 0 to skip.
                z2, if used, is always a feedforward SimpleRNN.
        inputs: input data, probably one of the outputs from `prepare_inputs`.
        zs_sample: Sample from q_f. Only required if using DSAE-Full.
        f_inputs_pre_z1: True if the zs_sample (if provided) joins as inputs to z1, otherwise it joins as inputs to z2.
        kernel_initializer: See tfkl RNN docs
        bias_initializer: See tfkl RNN docs
        recurrent_regularizer: See tfkl RNN docs

    Returns:
        A Tensor (or placeholder) with shape (samples (optional), batch_size, timesteps, units),
        where units refers to encd_rnn2_units if encd_rnn2_units > 0, else encd_rnn1_units.
    """
    if zs_sample is not None:
        # If zs_sample is a dist we need to transform it to a tensor.
        # Expand along time dimension by broadcast-add to zeros.
        n_times = tf.shape(inputs)[-2]
        zs_sample = zs_sample[..., tf.newaxis, :] + tf.zeros([n_times, 1])

    # Add optional f_input that we tile and concatenate onto _inputs.
    if zs_sample is not None and f_inputs_pre_z1:
        # Highly unlikely, but just in case inputs has samples dimension(s) then we can accommodate those here
        broadcast_shape_f = tf.concat((tf.shape(inputs)[:-3], [1, 1, 1]), 0)
        zs_sample = zs_sample + tf.zeros(broadcast_shape_f)

        # Expand inputs along sample dimension(s).
        broadcast_shape_inputs = tf.concat((tf.shape(zs_sample)[:-3], [1, 1, 1]), 0)
        inputs = inputs + tf.zeros(broadcast_shape_inputs)

        # Concatenate inputs with zs_sample
        inputs = tf.concat([inputs, zs_sample], axis=-1)  # (optional-samples, batch, timesteps, feat_dim+latent_static)

    z1_is_rnn = params['encd_rnn_type'].startswith('Bidirectional') \
        or (params['encd_rnn_type'] in ['GRU', 'LSTM', 'SimpleRNN', 'GRUClip'])
    has_z2 = 'encd_rnn2_units' in params and params['encd_rnn2_units'] > 0

    if z1_is_rnn or has_z2:
        rnn_kwargs = dict(
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            recurrent_regularizer=recurrent_regularizer,
            dropout=0,  # Dropout on inputs not needed.
            return_sequences=True)
        if params['encd_rnn_type'].endswith('GRU'):
            rnn_layer_cls = tfkl.GRU
        elif params['encd_rnn_type'].endswith('LSTM'):
            rnn_layer_cls = tfkl.LSTM
        elif params['encd_rnn_type'].endswith('SimpleRNN'):
            rnn_layer_cls = tfkl.SimpleRNN
        elif params['encd_rnn_type'].endswith('GRUClip'):
            rnn_layer_cls = GRUClip
            rnn_kwargs['clip_value'] = params['gru_clip_value']

    if z1_is_rnn:
        # Collapse samples + batch dims  -- required by LSTM
        sb_shape = tf.shape(inputs)[:-2]  # keep a record of the (samples,) batch shape.
        # new_shape = tf.concat(([-1], tf.shape(inputs)[-2:]), axis=0)  # Can't remember why I couldn't use -1 here.
        new_d1 = tf.reshape(tf.reduce_prod(tf.shape(inputs)[:-2]), (1,))
        new_shape = tf.concat((new_d1, tf.shape(inputs)[-2:]), 0)
        inputs = tf.reshape(inputs, new_shape)
        # inputs shape now (samples*batch, T, feat+lat_stat)

        if params['encd_rnn_type'].startswith('Bidirectional'):
            _enc_z = tfkl.Bidirectional(rnn_layer_cls(params['encd_rnn1_units'], **rnn_kwargs),
                                        merge_mode="concat", name="z_rnn_1")(inputs)
        else:
            _enc_z = rnn_layer_cls(params['encd_rnn1_units'], **rnn_kwargs)(inputs)

        # Restore leading samples, batch dims.
        _enc_z = tf.reshape(_enc_z, tf.concat((sb_shape, tf.shape(_enc_z)[1:]), axis=0))

    else:
        # Not RNN, just MLP
        _enc_z = tfkl.Dense(params['encd_rnn1_units'])(inputs)

    if params['zd_lag'] > 0:
        if params['encd_rnn_type'].startswith('Bidirectional'):
            # Shift _fwd back, dropping the latest samples, fill front with zeros
            # Shift _bwd forward, dropping the earliest samples, fill tail with zeros.
            # _fwd = [0,0,0,...,old_fwd[-lag:]]; _bwd = [old_bwd[lag:], ..., 0, 0, 0]
            _fwd, _bwd = tf.split(_enc_z, 2, axis=-1)
            _fwd = tf.concat([tf.zeros_like(_fwd[..., :params['zd_lag'], :]),
                              _fwd[..., :-params['zd_lag'], :]], axis=-2)
            _bwd = tf.concat([_bwd[..., params['zd_lag']:, :],
                              tf.zeros_like(_bwd[:, -params['zd_lag']:, :])], axis=-2)
            _enc_z = tf.concat([_fwd, _bwd], axis=-1)
        else:
            _enc_z = tf.concat([tf.zeros_like(_enc_z[..., :params['zd_lag'], :]),
                                _enc_z[..., :-params['zd_lag'], :]], axis=-2)

    if params['encd_rnn_type'].startswith('Bidirectional'):
        # Recombine forward and backward to get merge_mode="sum"
        _fwd, _bwd = tf.split(_enc_z, 2, axis=-1)
        _enc_z = _fwd + _bwd

    not_lfads = ('dec_rnn_type' not in params) or (params['dec_rnn_type'] != 'Complex')
    if not_lfads and has_z2:
        if zs_sample is not None and not f_inputs_pre_z1:
            # Highly unlikely, but just in case _enc_z has samples dimension(s) then we can accommodate those here
            broadcast_shape_f = tf.concat((tf.shape(_enc_z)[:-3], [1, 1, 1]), axis=0)
            zs_sample = zs_sample + tf.zeros(broadcast_shape_f)

            # Expand _enc_z along sample dimension(s).
            broadcast_shape_zenc = tf.concat((tf.shape(zs_sample)[:-3], [1, 1, 1]), axis=0)
            _enc_z = _enc_z + tf.zeros(broadcast_shape_zenc)

            # Concatenate _enc_z with zs_sample
            _enc_z = tf.concat([_enc_z, zs_sample],
                               axis=-1)  # (optional-samples, batch, timesteps, feat_dim+latent_static)

        # TODO: LFADS does an additional dropout before input to z2

        # Collapse samples + batch dims  -- required by RNNs
        sb_shape = tf.shape(_enc_z)[:-2]  # keep a record of the (samples,) batch shape.
        # new_shape = tf.concat(([-1], tf.shape(_enc_z)[-2:]), axis=0)  # Can't remember why I couldn't use -1 here.
        new_d1 = tf.reshape(tf.reduce_prod(tf.shape(_enc_z)[:-2]), (1,))
        new_shape = tf.concat((new_d1, tf.shape(_enc_z)[-2:]), axis=0)
        _enc_z = tf.reshape(_enc_z, new_shape)
        # _enc_z shape now (samples*batch, T, encd_rnn1_units+lat_stat)

        # z2 vanilla RNN used in DSAE Full. LFADS' z2 used elsewhere.
        _ = rnn_kwargs.pop('clip_value', None)
        _enc_z = tfkl.SimpleRNN(params['encd_rnn2_units'], **rnn_kwargs)(_enc_z)

        # Restore leading samples, batch dims.
        _enc_z = tf.reshape(_enc_z, tf.concat((sb_shape, tf.shape(_enc_z)[1:]), axis=0))

    return _enc_z


def make_encd_variational(params: dict, enc_z: tf.Tensor
                          ) -> Union[tfd.MultivariateNormalDiag, tfd.MultivariateNormalTriL]:
    """
    Take the encoded latent sequence z (output of z1 and optionally z2)
    and convert it to a distribution.

    This isn't necessary for LFADS models because z isn't in its final encoded
    form until inside the Complex cell, so it's up to the complex cell to
    handle the formation of the distribution.

    Args:
        params:
        - 'zd_size'
        - 'qzd_off_diag'
        - 'qzd_init_std'
        - 'q_samples'
        enc_z: input Tensor

    Returns:
        q_z: A tfd.Distribution.
        - q_z.sample() will not return a prepended samples dim if params['q_samples'] == 1, else it will.
        - q_z.sample(N) will always return a prepended samples dim (shape N), even if N == 1.
        If you need to reshape so the timesteps dim isn't considered in the "batch_shape" but is in the
        "event_shape", then you can use tfd.Independent(q_z, reinterpreted_batch_ndims=1).
    """
    from indl.dists import make_variational

    if 'dec_rnn_type' in params:
        assert params['dec_rnn_type'] != "Complex", "Skip this step. LFADS complex cell handles this intrinsically."

    # Get a multivariate normal diag over each timestep.
    q_z = make_variational(enc_z, params['zd_size'],
                           init_std=params['qzd_init_std'],
                           offdiag=params['qzd_off_diag'],
                           samps=params['q_samples'],
                           loc_name="z_loc", scale_name="z_scale",
                           use_mvn_diag=True)
    return q_z


def create_pzd(params: dict) -> IProcessMVNGenerator:
    """
    The z_prior is a sequence of multivariate diagonal normal distributions. The parameters of the distribution at
    each timestep are a function of (a sample drawn from) the distribution in the previous timestep.

    For DSAE, the process that governs the evolution of parameters over time is a RNN. For each trial, it is
    initialized to zeros and the first step is a zero-input. Subsequent inputs will be samples from the previous
    step. The RNN parameters are learnable. See process_dist.RNNMVNGenerator

    For LFADS, the process that governs the evolution of parameters over time is AR1. Each dimension is an
    independent process. The processes variances and the processes autocorrelation time constants (taus) are
    both trainable parameters. See process_dist.AR1ProcessMVNGenerator

    We also have process_dist.TiledMVNGenerator where a single distribution is shared over all timesteps.

    TODO: tfp also has Autoregressive and GaussianProcess distributions which can be parameterized with trainable
     variables and are maybe worth investigating here.

    The purpose of the prior is for KL-divergence loss, and I didn't have much luck using KL-divergence
    as a regularizer or model loss, so we will be calculating the KL-divergence during the manual train_step.
    It is therefore unnecessary to return a tensor-like here. Instead, we return an instance of a class, and that
    instance must have a .get_dist() method that we call explicitly during train_step to get a distribution,
    then the distribution will be used to calculate KL divergence.

    Args:
        params:
        - 'pzd_process': Which process to use for the sequence-of-MVN priors on z. Valid values:
            'AR1' - uses `AR1ProcessMVNGenerator`
            {rnn cell type} - including GRUClip, GRU, LSTM, RNN (not case-sensitive), uses `RNNMVNGenerator`
            'none' - uses `TiledMVNGenerator`

    Returns:
        an MVNGenerator.

        Get a sample and dist from the generator with
        `sample, dist = generator.get_dist(timestamps, samples=N_SAMPS, batch_size=BATCH_SIZE)`
        In most cases, when calculating KL and the returned dist is of the same type as the latent distribution,
        then the analytic KL can be calculated and only the dist event_shape matters, so samples and batch_size
        can be set to 1.
    """

    if params['pzd_process'] == 'AR1':
        from indl.dists.sequential import AR1ProcessMVNGenerator
        init_taus = [params['pzd_tau'] for _ in range(params['zd_size'])]
        init_std = [params['pzd_init_std'] for _ in range(params['zd_size'])]
        gen = AR1ProcessMVNGenerator(init_taus, init_std=init_std,
                                     trainable_mean=params['pzd_train_mean'],
                                     trainable_tau=params['pzd_train_tau'],
                                     trainable_var=params['pzd_train_var'],
                                     offdiag=params['pzd_offdiag'])

    elif params['pzd_process'] in ['RNN', 'LSTM', 'GRU', 'GRUClip']:
        from indl.dists.sequential import RNNMVNGenerator
        """
        units: int, out_dim: int, cell_type: str, shift_std: float = 0.1, offdiag: bool = False
        """
        gen = RNNMVNGenerator(params['pzd_units'], out_dim=params['zd_size'],
                              cell_type=params['pzd_process'],
                              shift_std=params['pzd_init_std'],
                              offdiag=params['pzd_offdiag'])

    else:
        from indl.dists.sequential import TiledMVNGenerator
        gen = TiledMVNGenerator(params['pzd_units'], init_std=params['pzd_init_std'],
                                trainable_mean=params['pzd_train_mean'],
                                trainable_var=params['pzd_train_var'],
                                offdiag=params['pzd_offdiag'])

    return gen


# TODO: Not even sure if this is necessary.
def sample_pzd(pzd: IProcessMVNGenerator, timesteps: int, params: dict, fixed=False)\
        -> Tuple[tf.Tensor, tfd.Independent]:
    """
    Samples from z_prior `timesteps` times.

    z_prior is a multivariate normal diagonal distribution for each timestep.
    We collect each timestep-dist's params (loc and scale), then use them to create
    the return value: a single MVN diag dist that has a dimension for timesteps.

    The cell returns a full dist for each timestep so that we can 'sample' it.
    If our sample size is 1, and our cell is an RNN cell, then this is roughly equivalent
    to doing a generative RNN (init state = zeros, return_sequences=True) then passing
    those values through a pair of Dense layers to parameterize a single MVNDiag.

    Args:
        gen: an instance of a concrete class that inherits from `indl.dists.sequential.IProcessMVNGenerator`,
            such as `AR1ProcessMVNGenerator`, `RNNMVNGenerator` or `TiledMVNGenerator`.
        timesteps: Number of timesteps to sample for each sequence.
        params:
            - q_samples
            - batch_size
        fixed: Boolean for whether or not to share the same random
            sample across all sequences in batch.

    Returns:
        A tuple of a sample from a distribution and the distribution itself.
        The tensor is of shape (samples, batch_size, timesteps, zd_size).
        The distribution is a tfd.Independent wrapping a multivariate normal diagonal.
    """
    return pzd.get_dist(timesteps, samples=params['q_samples'], batch_size=params['batch_size'], fixed=fixed)


def create_decoder(params: dict,
                   zs_sample: tf.Tensor,
                   enc_z: tf.Tensor,
                   ext_input=None,
                   kernel_initializer: str = 'lecun_normal',
                   bias_initializer: str = 'zeros',
                   recurrent_regularizer: str = 'l2'
                   )\
        -> Tuple[tf.Tensor, tf.Tensor]:
    """

    Args:
        params: a dict with keys. Please check 'generate_default_args' and 'generate_default_params' for
            definitive descriptions of each key. Required keys:
            'dec_rnn_type' - The cell type of the generator RNN.
            'gru_clip_value' - only required if 'dec_rnn_type' is (Bidirectional)GRUClip
            'dec_rnn_units' - number of units in the generator
            'zs_to_dec' - "initial conditions" or "tile inputs"
            'dropout_rate'
            'n_factors'
        zs_sample: A sample from q(f)
        enc_z: A sample from q(z_t)
        ext_input: Not supported
        kernel_initializer: passed to RNN cell
        bias_initializer: passed to RNN cell
        recurrent_regularizer: passed to RNN cell

    Returns:
        gen_outputs, factors
    """
    # Generate sequences and run through Dense layer, return factors

    if ext_input is not None:
        raise ValueError("Sorry, ext_input not supported yet.")

    if params['dec_rnn_type'].lower().startswith('complex'):
        raise ValueError("Please use create_generator_complex for complex cell.")

    # Other than LFADS, the other generator implementations are simply an RNN of the provided cell type.
    gen_is_rnn = params['dec_rnn_type'].startswith('Bidirectional') \
                 or (params['dec_rnn_type'] in ['GRU', 'LSTM', 'SimpleRNN', 'GRUClip'])
    assert gen_is_rnn, "dec_rnn_type must be a RNN cell type, " \
                       "possibly prefixed by 'Bidirectional'."

    rnn_kwargs = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        recurrent_regularizer=recurrent_regularizer,
        dropout=0,  # Dropout on inputs not needed.
        return_sequences=True)
    if params['dec_rnn_type'].endswith('GRU'):
        rnn_layer_cls = tfkl.GRU
    elif params['dec_rnn_type'].endswith('LSTM'):
        rnn_layer_cls = tfkl.LSTM
    elif params['dec_rnn_type'].endswith('SimpleRNN'):
        rnn_layer_cls = tfkl.SimpleRNN
    elif params['dec_rnn_type'].endswith('GRUClip'):
        rnn_layer_cls = GRUClip
        rnn_kwargs['clip_value'] = params['gru_clip_value']

    if params['dec_rnn_type'].startswith('Bidirectional'):
        rnn = tfkl.Bidirectional(rnn_layer_cls(params['dec_rnn_units'], **rnn_kwargs),
                                 merge_mode="concat", name="gen_rnn")
    else:
        rnn = rnn_layer_cls(params['dec_rnn_units'], **rnn_kwargs)

    #  The initial conditions are either a sample of q(z) or zeros. The inputs are either a sample of q(f),
    #  or a concatenation of a sample of q(f) and a tiling of a sample of q(z) (when initial conditions are zeros).
    #  Which input-formulation is used is in params['zs_to_dec']

    # Collapse samples + batch dims  -- required by LSTM
    # First for zs_sample
    sb_shape_f = tf.shape(zs_sample)[:-1]  # keep a record of the (samples,) batch shape.
    new_f_d1 = tf.reshape(tf.reduce_prod(sb_shape_f), (1,))
    new_f_shape = tf.concat((new_f_d1, tf.shape(zs_sample)[-1:]), 0)
    zs_sample = tf.reshape(zs_sample, new_f_shape)
    # --> zs_sample shape now (samples*batch, zs_size)
    # Next for enc_z (which is a sample for non-LFADS)
    sb_shape_z = tf.shape(enc_z)[:-2]  # keep a record of the (samples,) batch shape.
    new_z_d1 = tf.reshape(tf.reduce_prod(sb_shape_z), (1,))
    new_z_shape = tf.concat((new_z_d1, tf.shape(enc_z)[-2:]), 0)
    enc_z = tf.reshape(enc_z, new_z_shape)
    # --> enc_z shape now (samples*batch, timestamps, zd_size)

    if params['zs_to_dec'].lower().startswith('init'):
        _init_state = tfkl.Dense(params['dec_rnn_units'])(zs_sample)
        _gen_input = enc_z
    else:  # params['zs_to_dec'].lower().startswith('tile')
        _init_state = rnn.get_initial_state()  # This was trainable in LFADS!
        # Tile zs_sample over the timestamps dimension.
        dyn_steps = tf.shape(input=enc_z)[-2]
        _f = zs_sample[..., tf.newaxis, :] + tf.zeros([dyn_steps, 1])
        _gen_input = tf.concat([enc_z, _f], axis=-1)

    gen_outputs = rnn(_gen_input, initial_state=_init_state)
    # Restore samples dim with sb_shape
    restore_samples_shape = tf.concat((sb_shape_z, tf.shape(gen_outputs)[-2:]), 0)
    gen_outputs = tf.reshape(gen_outputs, restore_samples_shape)
    gen_dropped = tfkl.Dropout(params['dropout_rate'])(gen_outputs)
    factors = tfkl.Dense(params['n_factors'])(gen_dropped)

    return gen_outputs, factors


def create_decoder_complex(params: dict,
                           zs_sample: tf.Tensor,  # a sample from q(f)
                           z1: tf.Tensor,  # == z1 output
                           ext_input,  # Not implemented. Must be tensor (n_times, 0)
                           kernel_initializer: str = 'lecun_normal',
                           bias_initializer: str = 'zeros',
                           recurrent_regularizer: str = 'l2')\
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    if not params['dec_rnn_type'].lower().startswith('complex'):
        raise ValueError("Please use `create_generator` for non-complex cell")

    # LFADS' ComplexCell includes the z2 RNN, the generator RNN, and the to-factors Dense layer.
    # As the ComplexCell is run through a recurrent loop, it has a state similar to any RNN cell. However, the
    #  "complex state" includes the individual z2 RNN state and the generator RNN state. This can be confusing.
    # The initial "complex state" is zeros except the first portion which is a sample of q(f) provided in zs_sample.
    #  On this and subsequent steps, this part of the "complex state" containing zs_sample goes through a dropout
    #  layer and the to-factors Dense layer giving us prev_factors. prev_factors are then concatenated with z1 (z1
    #  is not a dist, obtained from z_enc input), run through dropout, and finally used as inputs to z2 RNN. The z2
    #  initial state is simply zeros. The z2 output is used to parameterize q(z_t).
    #  TODO: LFADS' z2 initial state is stored in a tf.Variable!
    # The generator inputs are a concatenation of a sample from q(z) and any external inputs if present. The
    #  generator initial state is the same sample from q(f) in f_enc used to calculate the prev_factors.
    # As a overly-simplified comparison with the other VAE formulations, we can say that the generator RNN
    #  gets its inputs from q(z) and its initial state from q(f).
    from indl.model.lfads.complex import ComplexCell
    custom_cell = ComplexCell(
        params['dec_rnn_units'],
        params['encd_rnn2_units'],
        params['n_factors'],
        params['zd_size'],
        params['ext_input_dim'],  # External input dimension.
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        recurrent_regularizer=recurrent_regularizer,
        dropout=params['dropout_rate'],
        clip_value=params['gru_clip_value']
    )
    complex_rnn = tfkl.RNN(custom_cell, return_sequences=True,
                           # recurrent_regularizer=tf.keras.regularizers.l2(l=gen_l2_reg),
                           name='complex_rnn')
    # Get RNN inputs
    ext_input_do = tfkl.Dropout(params['dropout_rate'])(ext_input)
    complex_input = tfkl.Concatenate()([z1, ext_input_do])
    # Get the RNN init states
    complex_init_states = complex_rnn.get_initial_state(complex_input)
    complex_init_states[0] = tfkl.Dense(params['dec_rnn_units'])(zs_sample)

    complex_output = complex_rnn(complex_input, initial_state=complex_init_states)
    gen_outputs, z2_state, z_latent_mean, z_latent_logvar, q_z_sample, factors = complex_output

    # We change the order on the output to match the vanilla `create_generator` output first 2 elements.
    return gen_outputs, factors, z2_state, z_latent_mean, z_latent_logvar, q_z_sample


def output_dist(factors: tf.Tensor, params: dict):

    if params['output_dist'].lower() == 'poisson':
        rates = tfkl.Dense(params['out_dim'])(factors)
        q_rates = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Poisson(rate=tf.exp(t)),
            name="rates_poiss"
        )(rates)
    else:
        from indl.dists import make_variational
        q_rates = make_variational(factors, params['out_dim'],
                                   init_std=params['qzs_init_std'],
                                   offdiag=params['output_dist_offdiag'],
                                   samps=1,
                                   loc_name="rates_loc", scale_name="rates_scale",
                                   use_mvn_diag=True)
    return q_rates

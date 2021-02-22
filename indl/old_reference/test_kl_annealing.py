import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors


# Constants #
#############
KL_MODE = 'loss_fun'  # 'regu', 'loss_fun'; 'addLossLayer' doesn't work.
PRIOR_TRAINABLE = True
BATCH_SIZE = 8
NDIM = 3
N_EPOCHS = 500


# Setup data #
##############
true_dist = tfd.MultivariateNormalDiag(
    loc=[-1., 1., 5],  # must have length == NDIM
    scale_diag=[0.5, 0.5, 0.9]
)
def gen_ds(n_iters=1e2):
    iter_ix = 0
    while iter_ix < n_iters:
        y_out = true_dist.sample()
        yield np.ones_like(y_out), y_out.numpy()
        iter_ix += 1
ds = tf.data.Dataset.from_generator(gen_ds, args=[1e2], output_types=(tf.float32, tf.float32),
                                    output_shapes=((NDIM,), (NDIM,))).batch(BATCH_SIZE)
if KL_MODE == 'loss_fun':
    ds = ds.map(lambda x, y: (x, (y, y)))


# Setup KL beta values #
########################
kl_beta = K.variable(value=0.0)
kl_beta._trainable = False  # It isn't trained. We set it explicitly with the callback.
def kl_beta_update(epoch_ix, N_epochs, M_cycles=5, R_increasing=0.8):
    T = N_epochs // M_cycles
    tau = (epoch_ix % T) / T
    new_beta_value = tf.minimum(1.0, tau / R_increasing)
    K.set_value(kl_beta, new_beta_value)
# Keras training callback
kl_beta_cb = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=lambda epoch, log: kl_beta_update(epoch, N_EPOCHS))

beta_testing = [
    {'beta': 0.0, 'name': 'b=0.'},
    {'beta': 1.0, 'name': 'b=1.'},
    {'beta': kl_beta, 'name': 'b=cycling'}
]


# Make Prior #
##############
def make_mvn_prior(ndim, trainable=False, offdiag=False):
    if trainable:
        scale_shift = np.log(np.exp(1) - 1).astype(np.float32)
        loc = tf.Variable(tf.random.normal([ndim], stddev=0.1, dtype=tf.float32))
        scale = tfp.util.TransformedVariable(
            tf.random.normal([ndim], mean=1.0, stddev=0.1, dtype=tf.float32),
            bijector=tfb.Chain([tfb.Shift(1e-5), tfb.Softplus(), tfb.Shift(scale_shift)]))
    else:
        loc = tf.zeros(ndim)
        scale = 1
    prior = tfd.Independent(tfd.Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1)
    return prior


def make_mvn_dist_fn(_x_, ndim):
    _loc = tfkl.Dense(ndim, name="loc_params")(_x_)
    _scale = tfkl.Dense(ndim, name="untransformed_scale_params")(_x_)
    _scale = tf.math.softplus(_scale + np.log(np.exp(1) - 1)) + 1e-5
    make_dist_fn = lambda t: tfd.Independent(tfd.Normal(loc=t[0], scale=t[1]))
    return make_dist_fn, [_loc, _scale]


# Setup Model(s) #
##################
def make_input_output(prior=None, kl_weight=None):
    _input = tfkl.Input(shape=(NDIM,))
    make_dist_fn, out_dist_params = make_mvn_dist_fn(_input, NDIM)
    if prior is not None:
        regu = tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=kl_weight)
    else:
        regu = None
    output = tfpl.DistributionLambda(
        name="out_dist",
        make_distribution_fn=make_dist_fn,
        activity_regularizer=regu
    )(out_dist_params)
    return _input, output


# Make and Train #
##################
neg_log_lik = lambda y_true, model_out: -model_out.log_prob(y_true)

loss_results = {}
dist_locs = []
for beta_dict in beta_testing:
    K.clear_session()
    tf.random.set_seed(42)
    # prior = tfd.MultivariateNormalDiag(loc=tf.zeros(NDIM), scale_diag=tf.ones(NDIM))
    prior = make_mvn_prior(NDIM, trainable=PRIOR_TRAINABLE)
    if KL_MODE == 'regu':
        _in, _out = make_input_output(prior=prior, kl_weight=beta_dict['beta'])
    else:
        _in, _out = make_input_output(prior=None, kl_weight=0.0)
        if KL_MODE == 'addLossLayer':
            raise ValueError("KLDivergenceAddLoss not working!")
            # Not actually a "passthrough", despite documentation
            # https://github.com/tensorflow/probability/issues/865
            # so we don't capture its output. But then it doesn't work!
            tfpl.KLDivergenceAddLoss(prior, weight=beta_dict['beta'])(_out)
    # Training
    if KL_MODE == 'loss_fun':
        model = tf.keras.Model(_in, (_out, _out))
        loss_fns = [neg_log_lik, lambda y_true, model_out: tfd.kl_divergence(model_out, prior)]
        loss_weights = [1.0, beta_dict['beta']]
    else:
        model = tf.keras.Model(_in, _out)
        loss_fns = neg_log_lik
        loss_weights = None
    model.compile(optimizer='adam', loss=loss_fns, loss_weights=loss_weights)
    hist = model.fit(ds, epochs=N_EPOCHS, verbose=2,
                     callbacks=[kl_beta_cb] if 'cyc' in beta_dict['name'] else None)
    loc_params = model.get_layer("loc_params").weights
    out_locs = np.ones((1, NDIM)) @ loc_params[0].numpy() + loc_params[1].numpy()
    print(f"Model {beta_dict['name']} est dist mean: "
          f"{out_locs}")
    print(f"prior mean: {prior.mean()}")
    loss_results[beta_dict['name']] = hist.history['loss']
    dist_locs.append((out_locs, prior.mean()))

for k, v in loss_results.items():
    plt.plot(v, label=k)
plt.xlabel("Epoch")
plt.ylabel("Loss (neg.log.lik)")
plt.legend()
plt.title(f"KL_MODE='{KL_MODE}'")
plt.show()

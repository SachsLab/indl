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
BATCH_SIZE = 8
N_EPOCHS = 5
PRIOR_TRAINABLE = True


# Setup data #
##############
true_dist = tfd.MultivariateNormalDiag(
    loc=[-1., 1., 5],  # must have length == NDIM
    scale_diag=[0.5, 0.5, 0.9]
)
NDIM = true_dist.event_shape[0]
def gen_ds(n_iters=1e2):
    iter_ix = 0
    while iter_ix < n_iters:
        y_out = true_dist.sample()
        yield np.ones((1,), dtype=np.float32), y_out.numpy()
        iter_ix += 1
ds = tf.data.Dataset.from_generator(gen_ds, args=[1e2], output_types=(tf.float32, tf.float32),
                                    output_shapes=((1,), (NDIM,))).batch(BATCH_SIZE)


def make_mvn_prior(ndim, trainable=False):
    if trainable:
        loc = tf.Variable(tf.random.normal([ndim], stddev=0.1, dtype=tf.float32),
                          name='prior_loc')
        scale = tfp.util.TransformedVariable(
            tf.random.normal([ndim], mean=1.0, stddev=0.1, dtype=tf.float32),
            bijector=tfb.Chain([tfb.Shift(1e-5), tfb.Softplus(), tfb.Shift(0.5413)]),
            name='prior_scale'
        )
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
def make_input_output(prior):
    _input = tfkl.Input(shape=(1,))
    make_dist_fn, dist_inputs = make_mvn_dist_fn(_input, NDIM)
    output = tfpl.DistributionLambda(
        name="out_dist",
        make_distribution_fn=make_dist_fn,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=0.1)
    )(dist_inputs)
    return _input, output


# Make and Train #
##################
K.clear_session()
tf.random.set_seed(42)
prior = make_mvn_prior(NDIM, trainable=PRIOR_TRAINABLE)
_in, _out = make_input_output(prior)
model = tf.keras.Model(_in, _out)
model.compile(optimizer='adam', loss=lambda y_true, model_out: -model_out.log_prob(y_true))
hist = model.fit(ds, epochs=N_EPOCHS, verbose=2)
loc_params = model.get_layer("loc_params").weights
out_locs = np.ones((1, 1)) @ loc_params[0].numpy() + loc_params[1].numpy()
print(f"Model est dist mean: {out_locs}")
print(f"prior mean: {prior.mean()}")


# Plot Loss #
#############
plt.plot(hist.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss (neg.log.lik)")
plt.show()

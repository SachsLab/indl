import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from indl.misc.sigfuncs import sigmoid
tfd = tfp.distributions
tfpl = tfp.layers


plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'lines.linewidth': 2,
    'lines.markersize': 5,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.figsize': (6.4, 6.4)
})



# Generate Toy Data

N_LATENTS = 4
N_SENSORS = 32
N_CLASSES = 5
FS = 64
DURATION = 2  # seconds
n_timesteps = int(DURATION * FS)

np.random.seed(66)
x = np.arange(n_timesteps) / FS
lat_freqs = np.random.uniform(low=0.5, high=2.5, size=(N_LATENTS,))
class_amps = np.random.uniform(low=-1.0, high=1.0, size=(N_CLASSES,))
mix_mat = np.random.randn(N_SENSORS, N_LATENTS)
mix_mat /= np.max(np.abs(mix_mat))

latent_protos = np.sin(lat_freqs[:, None]*2*np.pi*x[None, :])
f_sig = partial(sigmoid, B=5, x_offset=1.0)
latent_mods = class_amps[:, None] * f_sig(x)[None, :]
latent_class_dat = latent_mods[None, :, :] * latent_protos[:, None, :]
sensor_class_dat = mix_mat @ latent_class_dat.reshape(N_LATENTS, N_CLASSES * n_timesteps)
sensor_class_dat = sensor_class_dat.reshape(N_SENSORS, N_CLASSES, n_timesteps)
sensor_class_dat = np.transpose(sensor_class_dat, [1, 2, 0])


def draw_samples(class_idx, noise_std=0.01):
    dat = np.copy(sensor_class_dat[class_idx]).astype(np.float32)
    dat += np.random.normal(loc=0, scale=noise_std, size=dat.shape)
    return dat


N_TRIALS = 10000
BATCH_SIZE = 64

Y = np.random.randint(0, high=N_CLASSES, size=N_TRIALS)
X = draw_samples(Y)
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(BATCH_SIZE, drop_remainder=True)
print(dataset.element_spec)

# Specify priors
LATENT_SIZE_STATIC = 64
LATENT_SIZE_DYNAMIC = 16
HIDDEN_SIZE = 64
LATENT_POSTERIOR = "factorized"
BATCH_SIZE = 8
CLIP_NORM = 1e10
LEARNING_RATE = 0.0001
NUM_RECONSTRUCTION_SAMPLES = 1
NUM_SAMPLES = 4
RANDOM_SEED = 42
MAX_STEPS = 10000


# Inputs
_input = tfkl.Input(shape=(n_timesteps, N_SENSORS))

# Basic feature extractor (aka Compressor)
# Sprites example uses Conv2D to reduce from (batch x timesteps x 64 x 64) to (batch x timesteps x 1 x 1 x hidden_size)
# Maybe we can do this later.
_features = _input

# Static Encoder
# TODO: Check shapes
_x1 = tfkl.Bidirectional(tfkl.LSTM(HIDDEN_SIZE, return_sequences=False), merge_mode="sum")(_features)  # (batch, timesteps (1), hidden_size)
_loc = tfkl.Dense(LATENT_SIZE_STATIC)(_x1)  # latent variable means
_scale_diag = tfkl.Dense(LATENT_SIZE_STATIC, activation=tf.nn.softplus)(_x)  # Latent variable stds
_scale_diag = tf.math.add(_scale_diag, 1e-5)  # To keep > 0
_static_dist_params = tfkl.Concatenate()([_loc, _scale_diag])
_static_sample = tfpl.IndependentNormal(LATENT_SIZE_STATIC)(_static_dist_params)

# tfpl.DistributionLambda(
#     make_distribution_fn=lambda t: tfd.Normal(
#         loc=t[..., 0], scale=tf.exp(t[..., 1])),
#     convert_to_tensor_fn=lambda s: s.sample(5))

# TODO: Dynamic Encoder
# dist = EncoderDynamicFactorized(LATENT_SIZE_DYNAMIC, HIDDEN_SIZE)(inputs)
# samples = dist.sample(samples)
_x2 = tfkl.Concatenate()([_features, _static_sample])
_x2 = tfkl.Bidirectional(tfkl.LSTM(HIDDEN_SIZE, return_sequences=True), merge_mode="sum")(_x2)
_x2 = tfkl.SimpleRNN(HIDDEN_SIZE, return_sequences=True)(_x2)
_loc = tfkl.Dense(LATENT_SIZE_DYNAMIC)(_x2)  # latent variable means
_scale_diag = tfkl.Dense(LATENT_SIZE_DYNAMIC, activation=tf.nn.softplus)(_x2)  # Latent variable stds
_scale_diag = tf.math.add(_scale_diag, 1e-5)  # To keep > 0
_dynamic_dist_params = tfkl.Concatenate()([_loc, _scale_diag])
_dynamic_sample = tfpl.IndependentNormal(LATENT_SIZE_DYNAMIC)(_dynamic_dist_params)

# TODO: Decoder
# _latents = tfkl.Concatenate()([_dynamic_sample, _static_sample])
# _y = tfkl.Dense(?hidden, activation=tf.nn.leaky_relu)(_latents)
# _y = undo compressor
# likelihood = tfd.Indepdendent(distribution=tfd.Normal(loc=_y, scale=1.), reinterpreted_batch_ndims=3, name="decoded")

# Get Priors
from indl.model.tfp import LearnableMultivariateNormalDiag, LearnableMultivariateNormalDiagCell
# Static ... ?
static_prior = LearnableMultivariateNormalDiag(LATENT_SIZE_STATIC)()
# Dynamic
dynamic_prior = LearnableMultivariateNormalDiagCell(LATENT_SIZE_DYNAMIC, HIDDEN_SIZE)
dp_sample, dp_state = dynamic_prior.zero_state([NUM_SAMPLES, BATCH_SIZE])
locs = []
scale_diags = []
for _ in range(n_timesteps):
    dp_dist, dp_state = dynamic_prior(dp_sample, dp_state)
    dp_sample = dp_dist.sample()
    locs.append(dist.parameters["loc"])
    scale_diags.append(dist.parameters["scale_diag"])
loc = tf.stack(locs, axis=2)
scale_diag = tf.stack(scale_diags, axis=2)
dp_sample_dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

# Log prob of prior and posteriors given sample
static_prior_log_prob = static_prior.log_prob(static_sample)
static_posterior_log_prob = static_posterior.log_prob(static_sample)
dynamic_prior_log_prob = tf.reduce_sum(
    input_tensor=dynamic_prior.log_prob(dynamic_sample),
    axis=-1)  # sum time
dynamic_posterior_log_prob = tf.reduce_sum(
    input_tensor=dynamic_posterior.log_prob(dynamic_sample),
    axis=-1)  # sum time

#
likelihood_log_prob = tf.reduce_sum(
    input_tensor=likelihood.log_prob(inputs), axis=-1)  # sum time

# Collect log probs for elbo
elbo = tf.reduce_mean(input_tensor=static_prior_log_prob -
                                  static_posterior_log_prob +
                                  dynamic_prior_log_prob -
                                  dynamic_posterior_log_prob + likelihood_log_prob)
loss = -elbo

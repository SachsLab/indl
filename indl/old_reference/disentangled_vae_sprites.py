import functools
import tensorflow as tf
import tensorflow_probability as tfp
from absl import app
from absl import flags
import sprites_dataset

tfd = tfp.distributions
from indl.model.tfp import LearnableMultivariateNormalDiag, LearnableMultivariateNormalCell


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


static_prior = LearnableMultivariateNormalDiag(LATENT_SIZE_STATIC)
dynamic_prior = LearnableMultivariateNormalCell(HIDDEN_SIZE, LATENT_SIZE_DYNAMIC)


class Decoder(tf.keras.Model):
    """Probabilistic decoder for `p(x_t | z_t, f)`.

    The decoder generates a sequence of image frames `x_{1:T}` from
    dynamic and static latent variables `z_{1:T}` and `f`, respectively,
    for timesteps `1:T`.
    """

    def __init__(self, hidden_size, channels=3):
        """Constructs a probabilistic decoder.

        For each timestep, this model takes as input a concatenation of the
        dynamic and static latent variables `z_t` and `f`, respectively,
        outputs an intermediate representation via an affine function (i.e.,
        a one hidden layer MLP), then transforms this with four transpose
        convolution layers and up-sampling to the spatial shape of `x_t`.

        Together with the priors, this allows us to specify a generative
        model

        ```none
        p(x_{1:T}, z_{1:T}, f) = p(f) prod_{t=1}^T p(z_t | z_{<t}) p(x_t | z_t, f).
        ```

        Args:
          hidden_size: Dimensionality of the intermediate representations.
          channels: The depth of the output tensor.
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        activation = tf.nn.leaky_relu
        self.dense = tf.keras.layers.Dense(hidden_size, activation=activation)
        # Spatial sizes: (1,1) -> (8,8) -> (16,16) -> (32,32) -> (64,64).
        conv_transpose = functools.partial(
            tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
        self.conv_transpose1 = conv_transpose(256, 8, 1, padding="VALID")
        self.conv_transpose2 = conv_transpose(256, 3, 2)
        self.conv_transpose3 = conv_transpose(256, 3, 2)
        self.conv_transpose4 = conv_transpose(channels, 3, 2, activation=None)

    def call(self, inputs):
        """Runs the model to generate a distribution p(x_t | z_t, f).

        Args:
          inputs: A tuple of (z_{1:T}, f), where `z_{1:T}` is a tensor of
            shape [..., batch_size, timesteps, latent_size_dynamic], and `f`
            is of shape [..., batch_size, latent_size_static].

        Returns:
          A batched Independent distribution wrapping a set of Normal
          distributions over the pixels of x_t, where the Independent
          distribution has event shape [height, width, channels], batch
          shape [batch_size, timesteps], and sample shape [sample_shape,
          batch_size, timesteps, height, width, channels].
        """
        # We explicitly broadcast f to the same shape as z other than the final
        # dimension, because `tf.concat` can't automatically do this.
        dynamic, static = inputs
        timesteps = tf.shape(input=dynamic)[-2]
        static = static[..., tf.newaxis, :] + tf.zeros([timesteps, 1])
        latents = tf.concat([dynamic, static], axis=-1)  # (sample, N, T, latents)
        out = self.dense(latents)
        out = tf.reshape(out, (-1, 1, 1, self.hidden_size))
        out = self.conv_transpose1(out)
        out = self.conv_transpose2(out)
        out = self.conv_transpose3(out)
        out = self.conv_transpose4(out)  # (sample*N*T, h, w, c)
        expanded_shape = tf.concat(
            (tf.shape(input=latents)[:-1], tf.shape(input=out)[1:]), axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
        return tfd.Independent(
            distribution=tfd.Normal(loc=out, scale=1.),
            reinterpreted_batch_ndims=3,  # wrap (h, w, c)
            name="decoded_image")


class Compressor(tf.keras.Model):
    """Feature extractor.

    This convolutional model aims to extract features corresponding to a
    sequence of image frames for use in downstream probabilistic encoders.
    The architecture is symmetric to that of the convolutional decoder.
    """

    def __init__(self, hidden_size):
        """Constructs a convolutional compressor.

        This model takes as input `x_{1:T}` and outputs an intermediate
        representation for use in downstream probabilistic encoders.

        Args:
          hidden_size: Dimensionality of the intermediate representations.
        """
        super(Compressor, self).__init__()
        self.hidden_size = hidden_size
        # Spatial sizes: (64,64) -> (32,32) -> (16,16) -> (8,8) -> (1,1).
        conv = functools.partial(
            tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
        self.conv1 = conv(256, 3, 2)
        self.conv2 = conv(256, 3, 2)
        self.conv3 = conv(256, 3, 2)
        self.conv4 = conv(hidden_size, 8, padding="VALID")

    def call(self, inputs):
        """Runs the model to generate an intermediate representation of x_t.

        Args:
          inputs: A batch of image sequences `x_{1:T}` of shape
            `[sample_shape, batch_size, timesteps, height, width,
            channels]`.

        Returns:
          A batch of intermediate representations of shape [sample_shape,
          batch_size, timesteps, hidden_size].
        """
        image_shape = tf.shape(input=inputs)[-3:]
        collapsed_shape = tf.concat(([-1], image_shape), axis=0)
        out = tf.reshape(inputs, collapsed_shape)  # (sample*batch*T, h, w, c)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        expanded_shape = tf.concat((tf.shape(input=inputs)[:-3], [-1]), axis=0)
        return tf.reshape(out, expanded_shape)  # (sample, batch, T, hidden)


class EncoderStatic(tf.keras.Model):
    """Probabilistic encoder for the time-invariant latent variable `f`.

    The conditional distribution `q(f | x_{1:T})` is a multivariate
    normal distribution on `R^{latent_size}` at each timestep `t`,
    conditioned on intermediate representations of `x_{1:T}` from the
    convolutional encoder. The parameters are computed by passing the
    inputs through a bidirectional LSTM function, then passing the final
    output to an affine function to yield normal parameters for
    `q(f | x_{1:T})`.

    Together with the EncoderDynamicFactorized class, we can formulate the
    factorized approximate latent posterior `q` inference ("encoder")
    model as

    ```none
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) prod_{t=1}^T q(z_t | x_t).
    ```

    Together with the EncoderDynamicFull class, we can formulate the full
    approximate latent posterior `q` inference ("encoder") model as

    ```none
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) q(z_{1:T} | f, x_{1:T}).
    ```
    """

    def __init__(self, latent_size, hidden_size):
        """Constructs an encoder for `f`.

        Args:
          latent_size: An integer corresponding to the dimensionality of the
            distribution.
          hidden_size: Dimensionality of the LSTM, RNN, and affine function
            parameters.
        """
        super(EncoderStatic, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size),
            merge_mode="sum")
        self.output_layer = tf.keras.layers.Dense(2*latent_size)

    def call(self, inputs):
        """Runs the model to generate a distribution `q(f | x_{1:T})`.

        This generates a list of batched MultivariateNormalDiag
        distributions using the output of the recurrent model at each
        timestep to parameterize each distribution.

        Args:
          inputs: A batch of intermediate representations of image frames
            across all timesteps, of shape [..., batch_size, timesteps,
            hidden_size].

        Returns:
          A batched MultivariateNormalDiag distribution with event shape
          [latent_size], batch shape [..., batch_size], and sample shape
          [sample_shape, ..., batch_size, latent_size].
        """
        # TODO(dusenberrymw): Remove these reshaping commands after b/113126249 is
        # fixed.
        collapsed_shape = tf.concat(([-1], tf.shape(input=inputs)[-2:]), axis=0)
        out = tf.reshape(inputs, collapsed_shape)  # (sample*batch_size, T, hidden)
        out = self.bilstm(out)  # (sample*batch_size, hidden)
        expanded_shape = tf.concat((tf.shape(input=inputs)[:-2], [-1]), axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, batch_size, hidden)
        out = self.output_layer(out)  # (sample, batch_size, 2*latent_size)
        loc = out[..., :self.latent_size]
        scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5  # keep > 0
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class EncoderDynamicFactorized(tf.keras.Model):
    """Probabilistic encoder for the time-variant latent variable `z_t`.

    The conditional distribution `q(z_t | x_t)` is a multivariate normal
    distribution on `R^{latent_size}` at each timestep `t`, conditioned on
    an intermediate representation of `x_t` from the convolutional
    encoder. The parameters are computed by a one-hidden layer neural
    net.

    In this formulation, we posit that the dynamic latent variable `z_t`
    is independent of static latent variable `f`.

    Together with the EncoderStatic class, we can formulate the factorized
    approximate latent posterior `q` inference ("encoder") model as

    ```none
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) prod_{t=1}^T q(z_t | x_t).
    ```
    """

    def __init__(self, latent_size, hidden_size):
        """Constructs a "factorized" encoder for `z_t`.

        Args:
          latent_size: An integer corresponding to the
            dimensionality of the distribution.
          hidden_size: Dimensionality of the affine function parameters.
        """
        super(EncoderDynamicFactorized, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dense = tf.keras.layers.Dense(hidden_size, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(2*latent_size)

    def call(self, inputs):
        """Runs the model to generate a distribution `q(z_{1:T} | x_{1:T})`.

        Args:
          inputs: A batch of intermediate representations of image frames
            across all timesteps, of shape [..., batch_size, timesteps,
            hidden_size].

        Returns:
          A batch of MultivariateNormalDiag distributions with event shape
          [latent_size], batch shape [..., batch_size, timesteps], and
          sample shape [sample_shape, ..., batch_size, timesteps,
          latent_size].
        """
        out = self.dense(inputs)  # (..., batch, time, hidden)
        out = self.output_layer(out)  # (..., batch, time, 2*latent)
        loc = out[..., :self.latent_size]
        scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5  # keep > 0
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class EncoderDynamicFull(tf.keras.Model):
    """Probabilistic encoder for the time-variant latent variable `z_t`.

    The conditional distribution `q(z_{1:T} | x_{1:T}, f)` is a
    multivariate normal distribution on `R^{latent_size}` at each timestep
    `t`, conditioned on both an intermediate representation of the inputs
    `x_t` from the convolutional encoder, and on a sample of the static
    latent variable `f` at each timestep. The parameters are computed by
    passing the inputs through a bidirectional LSTM function, then passing
    these intermediates through an RNN function and an affine function to
    yield normal parameters for `q(z_t | x_{1:T}, f)`.

    In this formulation, we posit that `z_t` is conditionally dependent on
    `f`.

    Together with the EncoderStatic class, we can formulate the full
    approximate latent posterior `q` inference ("encoder") model as

    ```none
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) q(z_{1:T} | f, x_{1:T}).
    ```
    """

    def __init__(self, latent_size, hidden_size):
        """Constructs a "full" encoder for `z_t`.

        Args:
          latent_size: An integer corresponding to the
            dimensionality of the distribution.
          hidden_size: Dimensionality of the LSTM, RNN, and affine function
            parameters.
        """
        super(EncoderDynamicFull, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True),
            merge_mode="sum")
        self.rnn = tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True)
        self.output_layer = tf.keras.layers.Dense(2*latent_size)

    def call(self, inputs):
        """Runs the model to generate a distribution `q(z_{1:T} | x_{1:T}, f)`.

        This generates a list of batched MultivariateNormalDiag
        distributions using the output of the recurrent model at each
        timestep to parameterize each distribution.

        Args:
          inputs: A tuple of a batch of intermediate representations of
            image frames across all timesteps of shape [..., batch_size,
            timesteps, dimensions], and a sample of the static latent
            variable `f` of shape [..., batch_size, latent_size].

        Returns:
          A batch of MultivariateNormalDiag distributions with event shape
          [latent_size], batch shape [broadcasted_shape, batch_size,
          timesteps], and sample shape [sample_shape, broadcasted_shape,
          batch_size, timesteps, latent_size], where `broadcasted_shape` is
          the broadcasted sampled shape between the inputs and static
          sample.
        """
        # We explicitly broadcast `x` and `f` to the same shape other than the final
        # dimension, because `tf.concat` can't automatically do this. This will
        # entail adding a `timesteps` dimension to `f` to give the shape `(...,
        # batch, timesteps, latent)`, and then broadcasting the sample shapes of
        # both tensors to the same shape.
        features, static_sample = inputs
        length = tf.shape(input=features)[-2]
        static_sample = static_sample[..., tf.newaxis, :] + tf.zeros([length, 1])
        sample_shape_static = tf.shape(input=static_sample)[:-3]
        sample_shape_inputs = tf.shape(input=features)[:-3]
        broadcast_shape_inputs = tf.concat((sample_shape_static, [1, 1, 1]), 0)
        broadcast_shape_static = tf.concat((sample_shape_inputs, [1, 1, 1]), 0)
        features = features + tf.zeros(broadcast_shape_inputs)
        static_sample = static_sample + tf.zeros(broadcast_shape_static)
        # `combined` will have shape (..., batch, T, hidden+latent).
        combined = tf.concat((features, static_sample), axis=-1)
        # TODO(dusenberrymw): Remove these reshaping commands after b/113126249 is
        #  fixed.
        collapsed_shape = tf.concat(([-1], tf.shape(input=combined)[-2:]), axis=0)
        out = tf.reshape(combined, collapsed_shape)
        out = self.bilstm(out)  # (sample*batch, T, hidden_size)
        out = self.rnn(out)  # (sample*batch, T, hidden_size)
        expanded_shape = tf.concat(
            (tf.shape(input=combined)[:-2], tf.shape(input=out)[1:]), axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, batch, T, hidden_size)
        out = self.output_layer(out)  # (sample, batch, T, 2*latent_size)
        loc = out[..., :self.latent_size]
        scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5  # keep > 0
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class DisentangledSequentialVAE(tf.keras.Model):
    """Disentangled Sequential Variational Autoencoder.

    The disentangled sequential variational autoencoder posits a generative
    model in which a static, time-invariant latent variable `f` is sampled
    from a prior `p(f)`, a dynamic, time-variant latent variable `z_t` at
    timestep `t` is sampled from a conditional distribution
    `p(z_t | z_{<t})`, and an observation `x_t` is generated by a
    probabilistic decoder `p(x_t | z_t, f)`. The full generative model is
    defined as

    ```none
    p(x_{1:T}, z_{1:T}, f) = p(f) prod_{t=1}^T p(z_t | z_{<t}) p(x_t | z_t, f).
    ```

    We then posit an approximate posterior over the latent variables in the
    form of a probabilistic encoder `q(z_{1:T}, f | x_{1:T})`. Paired with
    the probabilistic decoder, we can form a sequential variational
    autoencoder model. Variational inference can be used to fit the model by
    decomposing the log marginal distribution `log p(x_{1:T})` into the
    evidence lower bound (ELBO) and the KL divergence between the true and
    approximate posteriors over the latent variables

    ```none
    log p(x) = -KL[q(z_{1:T},f|x_{1:T}) || p(x_{1:T},z_{1:T},f)]
             + KL[q(z_{1:T},f|x_{1:T}) || p(z_{1:T},f|x_{1:T})]
           = ELBO + KL[q(z_{1:T},f|x_{1:T}) || p(z_{1:T},f|x_{1:T})]
          >= ELBO  # Jensen's inequality for KL divergence.
          >= int int q(z_{1:T},f|x_{1:T}) [
              log p(x_{1:T},z_{1:T},f) - log q(z_{1:T},f|x_{1:T}) ] dz_{1:T} df.
    ```

    We then maximize the ELBO with respect to the model's parameters.

    The approximate posterior `q(z_{1:T}, f | x_{1:T})` can be formulated in
    two ways. The first formulation is a distribution that factorizes
    across timesteps,

    ```none
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) prod_{t=1}^T q(z_t | x_t),
    ```

    where `q(f | x_{1:T})` is a multivariate Gaussian parameterized by a
    bidirectional LSTM-based model, and `q(z_t | x_t)` is a multivariate
    Gaussian parameterized by a convolutional model. This is known as the
    "factorized" `q` distribution.

    The second formulation is a distribution

    ```none
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) q(z_{1:T} | f, x_{1:T}),
    ```

    where `q(z_{1:T} | f, x_{1:T})` is a multivariate Gaussian parameterized
    by a model consisting of a bidirectional LSTM followed by a basic RNN,
    and `q(f | x_{1:T})` is the same as previously described. This is known
    as the "full" `q` distribution.
    """

    def __init__(self, latent_size_static, latent_size_dynamic,
                   hidden_size, channels, latent_posterior):
        """Constructs a Disentangled Sequential Variational Autoencoder.

        Args:
          latent_size_static: Integer dimensionality of the static,
            time-invariant latent variable `f`.
          latent_size_dynamic: Integer dimensionality of each dynamic,
            time-variant latent variable `z_t`.
          hidden_size: Integer dimensionality of the model intermediates.
          channels: Integer depth of the output of the decoder.
          latent_posterior: Either "factorized" or "full" to indicate the
            formulation for the latent posterior `q`.
        """
        super(DisentangledSequentialVAE, self).__init__()
        self.latent_size_static = latent_size_static
        self.latent_size_dynamic = latent_size_dynamic
        self.hidden_size = hidden_size
        self.channels = channels
        self.latent_posterior = latent_posterior
        self.static_prior = LearnableMultivariateNormalDiag(latent_size_static)
        self.dynamic_prior = LearnableMultivariateNormalCell(
            latent_size_dynamic, hidden_size)
        self.decoder = Decoder(hidden_size, channels)
        self.compressor = Compressor(hidden_size)
        self.static_encoder = EncoderStatic(latent_size_static, hidden_size)
        if latent_posterior == "factorized":
            self.dynamic_encoder = EncoderDynamicFactorized(
                latent_size_dynamic, hidden_size)
        else:
            self.dynamic_encoder = EncoderDynamicFull(
                latent_size_dynamic, hidden_size)

    def generate(self, batch_size, length, samples=1, fix_static=False,
               fix_dynamic=False):
        """Generate new sequences.

        Args:
          batch_size: Number of sequences to generate.
          length: Number of timesteps to generate for each sequence.
          samples: Number of samples to draw from the latent distributions.
          fix_static: Boolean for whether or not to share the same random
            sample of the static latent variable `f` from its prior across
            all examples.
          fix_dynamic: Boolean for whether or not to share the same random
            sample of the dynamic latent variable `z_{1:T}` from its prior
            across all examples.

        Returns:
          A batched Independent distribution wrapping a set of Normal
          distributions over the pixels of the generated sequences, where
          the Independent distribution has event shape [height, width,
          channels], batch shape [samples, batch_size, timesteps], and
          sample shape [sample_shape, samples, batch_size, timesteps,
          height, width, channels].
        """
        static_sample, _ = self.sample_static_prior(samples, batch_size, fix_static)
        dynamic_sample, _ = self.sample_dynamic_prior(samples, batch_size, length,
                                                      fix_dynamic)
        likelihood = self.decoder((dynamic_sample, static_sample))
        return likelihood

    def reconstruct(self, inputs, samples=1, sample_static=False,
                  sample_dynamic=False, swap_static=False, swap_dynamic=False,
                  fix_static=False, fix_dynamic=False):
        """Reconstruct the given input sequences.

        Args:
          inputs: A batch of image sequences `x_{1:T}` of shape
            `[batch_size, timesteps, height, width, channels]`.
          samples: Number of samples to draw from the latent distributions.
          sample_static: Boolean for whether or not to randomly sample the
            static latent variable `f` from its prior distribution.
          sample_dynamic: Boolean for whether or not to randomly sample the
            dynamic latent variable `z_{1:T}` from its prior distribution.
          swap_static: Boolean for whether or not to swap the encodings for
            the static latent variable `f` between the examples.
          swap_dynamic: Boolean for whether or not to swap the encodings for
            the dynamic latent variable `z_{1:T}` between the examples.
          fix_static: Boolean for whether or not to share the same random
            sample of the static latent variable `f` from its prior across
            all examples.
          fix_dynamic: Boolean for whether or not to share the same random
            sample of the dynamic latent variable `z_{1:T}` from its prior
            across all examples.

        Returns:
          A batched Independent distribution wrapping a set of Normal
          distributions over the pixels of the reconstruction of the input,
          where the Independent distribution has event shape [height, width,
          channels], batch shape [samples, batch_size, timesteps], and
          sample shape [sample_shape, samples, batch_size, timesteps,
          height, width, channels].
        """
        batch_size = tf.shape(input=inputs)[-5]
        length = len(tf.unstack(inputs, axis=-4))  # hack for graph mode

        features = self.compressor(inputs)  # (..., batch, timesteps, hidden)

        if sample_static:
            static_sample, _ = self.sample_static_prior(
              samples, batch_size, fix_static)
        else:
            static_sample, _ = self.sample_static_posterior(features, samples)

        if swap_static:
            static_sample = tf.reverse(static_sample, axis=[1])

        if sample_dynamic:
            dynamic_sample, _ = self.sample_dynamic_prior(
                samples, batch_size, length, fix_dynamic)
        else:
            dynamic_sample, _ = self.sample_dynamic_posterior(
                features, samples, static_sample)

        if swap_dynamic:
            dynamic_sample = tf.reverse(dynamic_sample, axis=[1])

        likelihood = self.decoder((dynamic_sample, static_sample))
        return likelihood

    def sample_static_prior(self, samples, batch_size, fixed=False):
        """Sample the static latent prior.

        Args:
          samples: Number of samples to draw from the latent distribution.
          batch_size: Number of sequences to sample.
          fixed: Boolean for whether or not to share the same random
            sample across all sequences.

        Returns:
          A tuple of a sample tensor of shape [samples, batch_size,
          latent_size], and a MultivariateNormalDiag distribution from which
          the tensor was sampled, with event shape [latent_size], and batch
          shape [].
        """
        dist = self.static_prior()
        if fixed:  # in either case, shape is (samples, batch, latent)
            sample = dist.sample((samples, 1)) + tf.zeros([batch_size, 1])
        else:
            sample = dist.sample((samples, batch_size))
        return sample, dist

    def sample_static_posterior(self, inputs, samples):
        """Sample the static latent posterior.

        Args:
          inputs: A batch of intermediate representations of image frames
            across all timesteps, of shape [..., batch_size, timesteps,
            hidden_size].
          samples: Number of samples to draw from the latent distribution.

        Returns:
          A tuple of a sample tensor of shape [samples, batch_size,
          latent_size], and a MultivariateNormalDiag distribution from which
          the tensor was sampled, with event shape [latent_size], and batch
          shape [..., batch_size].
        """
        dist = self.static_encoder(inputs)
        sample = dist.sample(samples)
        return sample, dist

    def sample_dynamic_prior(self, samples, batch_size, length, fixed=False):
        """Sample the dynamic latent prior.

        Args:
          samples: Number of samples to draw from the latent distribution.
          batch_size: Number of sequences to sample.
          length: Number of timesteps to sample for each sequence.
          fixed: Boolean for whether or not to share the same random
            sample across all sequences in batch.

        Returns:
          A tuple of a sample tensor of shape [samples, batch_size, length
          latent_size], and a MultivariateNormalDiag distribution from which
          the tensor was sampled, with event shape [latent_size], and batch
          shape [samples, 1, length] if fixed or [samples, batch_size,
          length] otherwise.
        """
        if fixed:
            sample_batch_size = 1
        else:
            sample_batch_size = batch_size

        sample, state = self.dynamic_prior.zero_state([samples, sample_batch_size])
        locs = []
        scale_diags = []
        sample_list = []
        for _ in range(length):
            dist, state = self.dynamic_prior(sample, state)
            sample = dist.sample()
            locs.append(dist.parameters["loc"])
            scale_diags.append(dist.parameters["scale_diag"])
            sample_list.append(sample)

        sample = tf.stack(sample_list, axis=2)
        loc = tf.stack(locs, axis=2)
        scale_diag = tf.stack(scale_diags, axis=2)

        if fixed:  # tile along the batch axis
            sample = sample + tf.zeros([batch_size, 1, 1])

        return sample, tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    def sample_dynamic_posterior(self, inputs, samples, static_sample=None):
        """Sample the static latent posterior.

        Args:
          inputs: A batch of intermediate representations of image frames
            across all timesteps, of shape [..., batch_size, timesteps,
            hidden_size].
          samples: Number of samples to draw from the latent distribution.
          static_sample: A tensor sample of the static latent variable `f`
            of shape [..., batch_size, latent_size]. Only used
            for the full dynamic posterior formulation.

        Returns:
          A tuple of a sample tensor of shape [samples, batch_size, length
          latent_size], and a MultivariateNormalDiag distribution from which
          the tensor was sampled, with event shape [latent_size], and batch
          shape [broadcasted_shape, batch_size, length], where
          `broadcasted_shape` is the broadcasted sampled shape between the
          inputs and static sample.

        Raises:
          ValueError: If the "full" latent posterior formulation is being
            used, yet a static latent sample was not provided.
        """
        if self.latent_posterior == "factorized":
            dist = self.dynamic_encoder(inputs)
            samples = dist.sample(samples)  # (s, N, T, lat)
        else:  # full
            if static_sample is None:
                raise ValueError(
                    "The full dynamic posterior requires a static latent sample")
            dist = self.dynamic_encoder((inputs, static_sample))
            samples = dist.sample()  # (samples, N, latent)
        return samples, dist


def image_summary(seqs, name, num=None):
    """Visualizes sequences as TensorBoard summaries.

    Args:
    seqs: A tensor of shape [n, t, h, w, c].
    name: String name of this summary.
    num: Integer for the number of examples to visualize. Defaults to
      all examples.
    """
    seqs = tf.clip_by_value(seqs, 0., 1.)
    seqs = tf.unstack(seqs[:num])
    joined_seqs = [tf.concat(tf.unstack(seq), 1) for seq in seqs]
    joined_seqs = tf.expand_dims(tf.concat(joined_seqs, 0), 0)
    tf.compat.v2.summary.image(
      name,
      joined_seqs,
      max_outputs=1,
      step=tf.compat.v1.train.get_or_create_global_step())


def visualize_reconstruction(inputs, reconstruct, num=3, name="reconstruction"):
    """Visualizes the reconstruction of inputs in TensorBoard.

    Args:
    inputs: A tensor of the original inputs, of shape [batch, timesteps,
      h, w, c].
    reconstruct: A tensor of a reconstruction of inputs, of shape
      [batch, timesteps, h, w, c].
    num: Integer for the number of examples to visualize.
    name: String name of this summary.
    """
    reconstruct = tf.clip_by_value(reconstruct, 0., 1.)
    inputs_and_reconstruct = tf.concat((inputs[:num], reconstruct[:num]), axis=0)
    image_summary(inputs_and_reconstruct, name)


def visualize_qualitative_analysis(inputs, model, samples=1, batch_size=3,
                                   length=8):
    """Visualizes a qualitative analysis of a given model.

    Args:
    inputs: A tensor of the original inputs, of shape [batch, timesteps,
      h, w, c].
    model: A DisentangledSequentialVAE model.
    samples: Number of samples to draw from the latent distributions.
    batch_size: Number of sequences to generate.
    length: Number of timesteps to generate for each sequence.
    """
    average = lambda dist: tf.reduce_mean(
        input_tensor=dist.mean(), axis=0)  # avg over samples
    with tf.compat.v1.name_scope("val_reconstruction"):
        reconstruct = functools.partial(model.reconstruct, inputs=inputs,
                                        samples=samples)
        visualize_reconstruction(inputs, average(reconstruct()))
        visualize_reconstruction(inputs, average(reconstruct(sample_static=True)),
                                 name="static_prior")
        visualize_reconstruction(inputs, average(reconstruct(sample_dynamic=True)),
                                 name="dynamic_prior")
        visualize_reconstruction(inputs, average(reconstruct(swap_static=True)),
                                 name="swap_static")
        visualize_reconstruction(inputs, average(reconstruct(swap_dynamic=True)),
                                 name="swap_dynamic")

    with tf.compat.v1.name_scope("generation"):
        generate = functools.partial(model.generate, batch_size=batch_size,
                                     length=length, samples=samples)
        image_summary(average(generate(fix_static=True)), "fix_static")
        image_summary(average(generate(fix_dynamic=True)), "fix_dynamic")


def summarize_dist_params(dist, name, name_scope="dist_params"):
    """Summarize the parameters of a distribution.

    Args:
    dist: A Distribution object with mean and standard deviation
      parameters.
    name: The name of the distribution.
    name_scope: The name scope of this summary.
    """
    with tf.compat.v1.name_scope(name_scope):
        tf.compat.v2.summary.histogram(
            name="{}/{}".format(name, "mean"),
            data=dist.mean(),
            step=tf.compat.v1.train.get_or_create_global_step())
        tf.compat.v2.summary.histogram(
            name="{}/{}".format(name, "stddev"),
            data=dist.stddev(),
            step=tf.compat.v1.train.get_or_create_global_step())


def summarize_mean_in_nats_and_bits(inputs, units, name,
                                    nats_name_scope="nats",
                                    bits_name_scope="bits_per_dim"):
    """Summarize the mean of a tensor in nats and bits per unit.

    Args:
    inputs: A tensor of values measured in nats.
    units: The units of the tensor with which to compute the mean bits
      per unit.
    name: The name of the tensor.
    nats_name_scope: The name scope of the nats summary.
    bits_name_scope: The name scope of the bits summary.
    """
    mean = tf.reduce_mean(input_tensor=inputs)
    with tf.compat.v1.name_scope(nats_name_scope):
        tf.compat.v2.summary.scalar(
            name,
            mean,
            step=tf.compat.v1.train.get_or_create_global_step())
    with tf.compat.v1.name_scope(bits_name_scope):
        tf.compat.v2.summary.scalar(
            name,
            mean / units / tf.math.log(2.),
            step=tf.compat.v1.train.get_or_create_global_step())


def main(argv):
    del argv  # unused

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(RANDOM_SEED)

    sprites_data = sprites_dataset.SpritesDataset()

    model = DisentangledSequentialVAE(
      latent_size_static=LATENT_SIZE_STATIC,
      latent_size_dynamic=LATENT_SIZE_DYNAMIC,
      hidden_size=HIDDEN_SIZE, channels=sprites_data.channels,
      latent_posterior=LATENT_POSTERIOR)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.compat.v1.train.AdamOptimizer(
        tf.compat.v1.train.cosine_decay(LEARNING_RATE, global_step, MAX_STEPS))

    dataset = sprites_data.train.map(lambda *x: x[0]).shuffle(1000).repeat()
    dataset = dataset.batch(BATCH_SIZE).take(MAX_STEPS)

    for inputs in dataset.prefetch(buffer_size=None):

        with tf.GradientTape() as tape:
            features = model.compressor(inputs)  # (batch, timesteps, hidden)
            static_sample, static_posterior = model.sample_static_posterior(
                features, NUM_SAMPLES)  # (samples, batch, latent)
            dynamic_sample, dynamic_posterior = model.sample_dynamic_posterior(
                features, NUM_SAMPLES, static_sample)  # (samples, N, T, latent)
            likelihood = model.decoder((dynamic_sample, static_sample))

            reconstruction = tf.reduce_mean(  # integrate samples
                input_tensor=likelihood.mean()[:NUM_RECONSTRUCTION_SAMPLES],
                axis=0)
            visualize_reconstruction(inputs, reconstruction,
                                     name="train_reconstruction")

            static_prior = model.static_prior()
            _, dynamic_prior = model.sample_dynamic_prior(
                NUM_SAMPLES, BATCH_SIZE, sprites_data.length)

            summarize_dist_params(static_prior, "static_prior")
            summarize_dist_params(static_posterior, "static_posterior")
            summarize_dist_params(dynamic_prior, "dynamic_prior")
            summarize_dist_params(dynamic_posterior, "dynamic_posterior")
            summarize_dist_params(likelihood, "likelihood")

            static_prior_log_prob = static_prior.log_prob(static_sample)
            static_posterior_log_prob = static_posterior.log_prob(static_sample)
            dynamic_prior_log_prob = tf.reduce_sum(
                input_tensor=dynamic_prior.log_prob(dynamic_sample),
                axis=-1)  # sum time
            dynamic_posterior_log_prob = tf.reduce_sum(
                input_tensor=dynamic_posterior.log_prob(dynamic_sample),
                axis=-1)  # sum time
            likelihood_log_prob = tf.reduce_sum(
                input_tensor=likelihood.log_prob(inputs), axis=-1)  # sum time


            with tf.compat.v1.name_scope("log_probs"):
                summarize_mean_in_nats_and_bits(
                    static_prior_log_prob, LATENT_SIZE_STATIC,
                    "static_prior")
                summarize_mean_in_nats_and_bits(
                    static_posterior_log_prob, LATENT_SIZE_STATIC,
                    "static_posterior")
                summarize_mean_in_nats_and_bits(
                    dynamic_prior_log_prob, LATENT_SIZE_DYNAMIC * sprites_data.length,
                    "dynamic_prior")
                summarize_mean_in_nats_and_bits(
                    dynamic_posterior_log_prob, LATENT_SIZE_DYNAMIC * sprites_data.length,
                    "dynamic_posterior")
                summarize_mean_in_nats_and_bits(
                    likelihood_log_prob, sprites_data.frame_size ** 2 * sprites_data.channels * sprites_data.length,
                    "likelihood")

            elbo = tf.reduce_mean(input_tensor=static_prior_log_prob -
                                  static_posterior_log_prob +
                                  dynamic_prior_log_prob -
                                  dynamic_posterior_log_prob + likelihood_log_prob)
            loss = -elbo
            tf.compat.v2.summary.scalar(
                "elbo",
                elbo,
                step=tf.compat.v1.train.get_or_create_global_step())

        grads = tape.gradient(loss, model.variables)
        grads, global_norm = tf.clip_by_global_norm(grads, CLIP_NORM)
        grads_and_vars = list(zip(grads, model.variables))  # allow reuse in py3

        with tf.compat.v1.name_scope("grads"):
            tf.compat.v2.summary.scalar(
                "global_norm_grads",
                global_norm,
                step=tf.compat.v1.train.get_or_create_global_step())
            tf.compat.v2.summary.scalar(
                "global_norm_grads_clipped",
                tf.linalg.global_norm(grads),
                step=tf.compat.v1.train.get_or_create_global_step())

        for grad, var in grads_and_vars:
            with tf.compat.v1.name_scope("grads"):
                tf.compat.v2.summary.histogram(
                    "{}/grad".format(var.name),
                    data=grad,
                    step=tf.compat.v1.train.get_or_create_global_step())
            with tf.compat.v1.name_scope("vars"):
                tf.compat.v2.summary.histogram(
                    var.name,
                    data=var,
                    step=tf.compat.v1.train.get_or_create_global_step())

        optimizer.apply_gradients(grads_and_vars, global_step)

    if global_step.numpy() == MAX_STEPS:
        print("ELBO ({}/{}): {}".format(global_step.numpy(), MAX_STEPS, elbo.numpy()))
        with tf.compat.v2.summary.record_if(True):
            val_data = sprites_data.test.take(20)
            inputs = next(iter(val_data.shuffle(20).batch(3)))[0]
            visualize_qualitative_analysis(inputs, model, NUM_RECONSTRUCTION_SAMPLES)


if __name__ == "__main__":
    app.run(main)

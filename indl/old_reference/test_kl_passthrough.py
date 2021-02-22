import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers


N_LATENT = 5
prior = tfd.Independent(tfd.Normal(loc=np.zeros((N_LATENT,), dtype=np.float32),
                                   scale=np.ones((N_LATENT,), dtype=np.float32)),
                        reinterpreted_batch_ndims=1)

input = tfkl.Input(shape=(16,))
x1 = tfkl.Dense(N_LATENT * 2)(input)
x2 = tfpl.IndependentNormal(event_shape=(N_LATENT,))(x1)
x3 = tfpl.KLDivergenceAddLoss(prior, use_exact_kl=True)(x2)
print(x2)  # --> tfp.distributions.Independent
print(x3)  # --> Tensor

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers as tfkl


# diag_gaussian_log_likelihood
# gaussian_pos_log_likelihood
# Gaussian
# DiagonalGaussianFromExisting
# LearnableDiagonalGaussian
# LearnableAutoRegressive1Prior
# KLCost_GaussianGaussianProcessSampled


def test_ar1_prior():
    from indl.model.lfads.dists import LearnableAutoRegressive1Prior

    # TODO: ar1 tests
    pass
    """
    autocorrelation_taus = [hps.prior_ar_atau for _ in range(hps.co_dim)]
    noise_variances = [hps.prior_ar_nvar for _ in range(hps.co_dim)]
    cos_prior = LearnableAutoRegressive1Prior(graph_batch_size, hps.co_dim,
                                              autocorrelation_taus,
                                              noise_variances,
                                              hps.do_train_prior_ar_atau,
                                              hps.do_train_prior_ar_nvar,
                                              "u_prior_ar1")
    kl_cost_co_b_t = KLCost_GaussianGaussianProcessSampled(cos_posterior, cos_prior).kl_cost_b
    """

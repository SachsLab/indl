import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from indl.misc.kernels import sskernel
from indl.misc.kernels import Gauss
from indl.misc.kernels import Boxcar
from indl.misc.kernels import Alpha
from indl.misc.kernels import Exponential


class TestKernels:
    srate = 1000
    spk_rate = 13.0  # Avg 30 spikes per second
    tvec = np.arange(srate) / srate
    np.random.seed(1337)
    spikeevents = (np.random.rand(srate) < (spk_rate / srate)).astype(np.float32)
    spiketimes = tvec[spikeevents.nonzero()]

    def test_sskernel(self):
        kernel_width = sskernel(self.spiketimes - self.spiketimes[0], nbs=0)[2]
        assert np.abs(kernel_width - 0.12930384) < 0.001, "Unexpected kernel_width returned"

    def test_gauss(self):
        kernel_width = sskernel(self.spiketimes - self.spiketimes[0], nbs=0)[2]
        kernel_param = 1 / (2.0 * 2.7) * kernel_width
        span_fac = 3.0
        t_kern = np.arange(-span_fac * kernel_param, span_fac * kernel_param + (1 / self.srate), 1 / self.srate)
        kernel = Gauss(t_kern, kernel_param)
        assert kernel.size == len(t_kern)
        assert np.all(kernel >= 0)

    def test_boxcar(self):
        kernel_param = 0.05  # The width of the rectangle in seconds
        span_fac = np.sqrt(3.0)
        kernel_param /= (2 * np.sqrt(3.0))
        t_kern = np.arange(-span_fac * kernel_param, span_fac * kernel_param + (1 / self.srate), 1 / self.srate)
        kernel = Boxcar(t_kern, kernel_param)
        assert kernel[0] == kernel[-1] == 0
        assert np.var(kernel[1:-1]) == 0

    def test_alpha(self):
        kernel_param = 0.03  # tau
        kernel_param *= np.sqrt(2)
        span_fac = 6.0
        t_kern = np.arange(-span_fac * kernel_param, span_fac * kernel_param + (1 / self.srate), 1 / self.srate)
        kernel = Alpha(t_kern, kernel_param)
        assert np.all(kernel[:kernel.size//2] == 0)
        assert np.var(kernel[kernel.size // 2:]) > 0

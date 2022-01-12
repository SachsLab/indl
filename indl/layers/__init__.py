import numbers
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.eager import context


class CoordinatedDropout(tfkl.Dropout):
    """
    Applies Dropout to the input.
    Its call returns a tuple of the dropped inputs and the mask indicating dropped features.
    """

    def compute_output_shape(self, input_shape):
        return input_shape, input_shape

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            rate = self.rate
            noise_shape = self.noise_shape
            seed = self.seed
            with ops.name_scope(None, "coordinated_dropout", [inputs]) as name:
                is_rate_number = isinstance(rate, numbers.Real)
                if is_rate_number and (rate < 0 or rate >= 1):
                    raise ValueError("rate must be a scalar tensor or a float in the "
                                     "range [0, 1), got %g" % rate)
                x = ops.convert_to_tensor(inputs, name="x")
                x_dtype = x.dtype
                if not x_dtype.is_floating:
                    raise ValueError("x has to be a floating point tensor since it's going "
                                     "to be scaled. Got a %s tensor instead." % x_dtype)
                is_executing_eagerly = context.executing_eagerly()
                if not tensor_util.is_tensor(rate):
                    if is_rate_number:
                        keep_prob = 1 - rate
                        scale = 1 / keep_prob
                        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
                        ret = gen_math_ops.mul(x, scale)
                    else:
                        raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)
                else:
                    rate.get_shape().assert_has_rank(0)
                    rate_dtype = rate.dtype
                    if rate_dtype != x_dtype:
                        if not rate_dtype.is_compatible_with(x_dtype):
                            raise ValueError(
                                "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
                                (x_dtype.name, rate_dtype.name, rate))
                        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
                    one_tensor = constant_op.constant(1, dtype=x_dtype)
                    ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

                noise_shape = nn_ops._get_noise_shape(x, noise_shape)
                # Sample a uniform distribution on [0.0, 1.0) and select values larger
                # than rate.
                #
                # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
                # and subtract 1.0.
                random_tensor = random_ops.random_uniform(
                    noise_shape, seed=seed, dtype=x_dtype)
                # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
                # hence a >= comparison is used.
                keep_mask = random_tensor >= rate
                ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
                if not is_executing_eagerly:
                    ret.set_shape(x.get_shape())
                return ret, keep_mask

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: (array_ops.identity(inputs), array_ops.ones_like(inputs) > 0))
        return output

__all__ = ['AutoShapeMixin']


from typing import Any, Mapping, Optional, Sequence, TypeVar, Union

import tensorflow as tf
import tensorflow.keras.layers as tfkl


T = TypeVar('T')
Nested = Union[T, Sequence[T], Mapping[Any, T]]


class AutoShapeMixin:
    """
    Mixin for `tf.keras.layers.Layer`s and subclasses to automatically define input and output specs the first time the model is called. Must be listed before `tf.keras.layers.Layer` when subclassing. Only works for
    models and layers with static input and output shapes. First `batch_dims` dimensions (default 1) are assumed to be batch dimensions.
    https://gist.github.com/Danmou/bafa5c80356fdb2c843eaf38c8597f84
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.batch_dims: int = kwargs.pop('batch_dims', 1)
        super().__init__(*args, **kwargs)
        assert not getattr(self, 'dynamic'), 'AutoShapeMixin should not be used with dynamic layers!'
        self._input_spec: Optional[Nested[tfkl.InputSpec]] = None
        self._output_spec: Optional[Nested[tfkl.InputSpec]] = None
        self.built_with_input = False

    def build_with_input(self, input: Nested[tf.Tensor], *args: Any, **kwargs: Any) -> None:
        bd = self.batch_dims
        self._input_spec = tf.nest.map_structure(
            lambda x: tfkl.InputSpec(shape=[None]*bd + x.shape[bd:], dtype=x.dtype), input)
        dummy_input = tf.nest.map_structure(lambda t: tf.zeros([2]*bd + t.shape[bd:], t.dtype), input)
        dummy_output = super().__call__(dummy_input, *args, **kwargs)
        self._output_spec = tf.nest.map_structure(lambda x: tfkl.InputSpec(shape=[None]*bd + x.shape[bd:],
                                                                             dtype=x.dtype), dummy_output)
        self.built_with_input = True

    def __call__(self, inputs: Nested[tf.Tensor], *args: Any, **kwargs: Any) -> Any:
        if not self.built_with_input:
            self.build_with_input(inputs, *args, **kwargs)
        return super().__call__(inputs, *args, **kwargs)

    @property
    def input_spec(self) -> Optional[Nested[tfkl.InputSpec]]:
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: Optional[tfkl.InputSpec]) -> None:
        self._input_spec = value

    @property
    def output_spec(self) -> Optional[Nested[tfkl.InputSpec]]:
        return self._output_spec

    @output_spec.setter
    def output_spec(self, value: Optional[tfkl.InputSpec]) -> None:
        self._output_spec = value

    @property
    def input_shape(self) -> Nested[tf.TensorShape]:
        assert self.input_spec is not None, 'build_with_input has not been called; input shape is not defined'
        return tf.nest.map_structure(lambda x: x.shape, self.input_spec)

    @property
    def output_shape(self) -> Nested[tf.TensorShape]:
        assert self.output_spec is not None, 'build_with_input has not been called; output shape is not defined'
        return tf.nest.map_structure(lambda x: x.shape, self.output_spec)

    @property
    def input_dtype(self) -> Nested[tf.TensorShape]:
        assert self.input_spec is not None, 'build_with_input has not been called; input dtype is not defined'
        return tf.nest.map_structure(lambda x: x.dtype, self.input_spec)

    @property
    def output_dtype(self) -> Nested[tf.TensorShape]:
        assert self.output_spec is not None, 'build_with_input has not been called; output dtype is not defined'
        return tf.nest.map_structure(lambda x: x.dtype, self.output_spec)

    def compute_output_shape(self, input_shape: Nested[tf.TensorShape]) -> Nested[tf.TensorShape]:
        if self.output_spec is None:
            return super().compute_output_shape(input_shape)
        batch_shape = tf.nest.flatten(input_shape)[0][:self.batch_dims]
        return tf.nest.map_structure(lambda x: batch_shape + x[self.batch_dims:], self.output_shape)

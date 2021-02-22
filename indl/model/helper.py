__all__ = ['check_inputs', 'check_tensor_name']


import tensorflow as tf


def check_inputs(model_builder):
    def _check_inputs_wrapped_model_builder(*args, **kwargs):
        _input = args[0]

        is_multi = isinstance(_input, (list, tuple)) and\
                   len(_input) > 0 and not any([isinstance(_, int) for _ in _input])

        def to_input_tensor(single_spec):
            if isinstance(single_spec, dict):
                _in = {name: tf.keras.layers.Input(shape=spec.shape[1:], name=name) for name, spec in single_spec.items()}
            elif isinstance(single_spec, (tf.TensorSpec, tf.Tensor)):
                _in = tf.keras.layers.Input(shape=single_spec.shape[1:])
            elif isinstance(single_spec, (tuple, tf.TensorShape)):
                _in = tf.keras.layers.Input(shape=single_spec)
                # TODO: If we want single_spec to have batch dimension then change above to shape=single_spec[1:]
            else:
                raise ValueError("tf Dataset input must be laid out as a tuple, dict, or tensor.")
            return _in

        if is_multi:
            _input = [to_input_tensor(_) for _ in _input]
        elif isinstance(_input, tf.Tensor):
            if 'return_model' not in kwargs or kwargs['return_model'] is None:
                kwargs['return_model'] = False
            elif kwargs['return_model'] is True:
                _input = to_input_tensor(_input)
        else:
            _input = to_input_tensor(_input)

        args = (_input,) + args[1:]
        return model_builder(*args, **kwargs)

    return _check_inputs_wrapped_model_builder


def check_tensor_name(t, name):
    """
    We need a try, except block to check tensor names because
    regu loss tensors do not allow access to their .name in eager mode.
    """
    try:
        return name in t.name
    except AttributeError:
        return False

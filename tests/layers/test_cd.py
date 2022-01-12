import tensorflow as tf
from tensorflow.keras import backend as K


def test_coordinated_dropout():
    from indl.layers import CoordinatedDropout

    K.clear_session()
    n_times = None
    n_sensors = 36
    inputs = tf.keras.Input(shape=(n_times, n_sensors))
    dropped, cd_mask = CoordinatedDropout(0.5)(inputs)
    assert dropped.shape[0] == None and dropped.shape[1] == n_times and dropped.shape[2] == n_sensors
    assert all([dropped.shape[_] == cd_mask.shape[_] for _ in range(len(dropped.shape))])
    assert cd_mask.dtype == tf.bool
    # Test it in a Keras Model
    test_model = tf.keras.Model(inputs=inputs, outputs=[dropped, cd_mask])
    batch_size = 16
    n_times = n_times or 2
    # First during training
    K.set_learning_phase(1)
    dropped, cd_mask = test_model(tf.random.uniform((batch_size, n_times, n_sensors)))
    assert cd_mask.numpy().sum() < cd_mask.numpy().size, "CD mask during training should have some 0 elements"
    # Then outside training
    K.set_learning_phase(0)
    dropped, cd_mask = test_model(tf.random.uniform((batch_size, n_times, n_sensors)))
    assert cd_mask.numpy().sum() == cd_mask.numpy().size, "CD mask outside training should be all 1s"

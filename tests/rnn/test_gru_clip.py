import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers as tfkl


def test_GRUClipCell():
    from indl.rnn.gru_clip import GRUClipCell

    K.clear_session()

    n_times, n_sensors = 246, 36
    batch_size = 16
    f_units = 128

    f_enc_inputs = tf.keras.Input(shape=(n_times, n_sensors))
    cell = GRUClipCell(f_units)
    assert isinstance(cell, tfkl.GRUCell)
    assert cell.units == f_units
    init_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    assert init_state.shape.as_list() == [batch_size, f_units]
    rnn = tfkl.RNN(cell)
    assert rnn.cell == cell
    bidir = tfkl.Bidirectional(rnn)
    final_state = bidir(f_enc_inputs)
    assert final_state.shape.as_list()[-1] == (f_units * 2)

    model = tf.keras.Model(inputs=f_enc_inputs, outputs=final_state, name="GRUClip")
    dummy_state = model(tf.random.uniform((batch_size, n_times, n_sensors)))
    assert dummy_state.shape.as_list() == [batch_size, f_units * 2]
    assert (dummy_state.numpy() != 0).sum() > 0


def test_GRUClipLayer():
    from indl.rnn.gru_clip import GRUClip

    K.clear_session()

    n_times, n_sensors = 246, 36
    batch_size = 16
    f_units = 128

    f_enc_inputs = tf.keras.Input(shape=(n_times, n_sensors))
    final_state = GRUClip(f_units)(f_enc_inputs)

    model = tf.keras.Model(inputs=f_enc_inputs, outputs=final_state, name="GRUClip")
    # model.summary()

    dummy_state = model(tf.random.uniform((batch_size, n_times, n_sensors)))
    # print(dummy_state)


def test_GRUClipLayer_in_Bidirectional():
    from indl.rnn.gru_clip import GRUClip

    K.clear_session()

    n_times, n_sensors = 246, 36
    batch_size = 16
    f_units = 128

    f_enc_inputs = tf.keras.Input(shape=(n_times, n_sensors))

    rnn_layer = GRUClip(f_units)

    bd_layer = tfkl.Bidirectional(rnn_layer, merge_mode="concat")
    final_state = bd_layer(f_enc_inputs)

    model = tf.keras.Model(inputs=f_enc_inputs, outputs=final_state, name="GRUClipBi")
    # model.summary()

    dummy_state = model(tf.random.uniform((batch_size, n_times, n_sensors)))
    # print(dummy_state)
    # TODO: Test something about dummy_state

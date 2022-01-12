import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as K


def test_ComplexCell():
    # TODO: Convert to using parameter names from parent beta_vae, including f, z1, z2
    from indl.model.lfads.complex import ComplexCell
    K.clear_session()

    f_latent_size = 10
    gen_n_hidden = 100
    z2_units = 30  # LFADS' con_hidden_state_dim
    n_factors = 10
    z_latent_size = 12  # LFADS' co_dim
    ext_input_dim = 0

    z_enc_dim = 64
    timesteps = 300
    batch_size = 16
    gen_l2_reg = 0.01

    # Create placeholders
    # Three external pieces of data coming into the generator.
    f_enc = tf.keras.Input(shape=(f_latent_size,), name="f_enc")
    z1_enc = tf.keras.Input(shape=(timesteps, z_enc_dim), name="z_enc")
    ext_input = tf.keras.Input(shape=(timesteps, ext_input_dim), name="ext_input")

    custom_cell = ComplexCell(
        gen_n_hidden,
        z2_units,
        n_factors,
        z_latent_size,
        ext_input_dim,
    )

    # Create the RNN.
    complex_rnn = tfkl.RNN(custom_cell, return_sequences=True,
                           # recurrent_regularizer=tf.keras.regularizers.l2(l=gen_l2_reg),
                           name='complex_rnn')

    # Get the RNN inputs
    ext_input_do = tfkl.Dropout(0.01)(ext_input)
    complex_input = tfkl.Concatenate()([z1_enc, ext_input_do])
    # Get the RNN init states
    complex_init_states = complex_rnn.get_initial_state(complex_input)
    # Replace init_states[0] with encoded f, Dense to make same size as init states
    complex_init_states[0] = tfkl.Dense(gen_n_hidden)(f_enc)

    # Run placeholders through RNN
    #
    complex_output = complex_rnn(complex_input, initial_state=complex_init_states)
    # Build the model.
    generator_model = tf.keras.Model(inputs=[f_enc, z1_enc, ext_input], outputs=complex_output)
    print(generator_model.summary())

    dummy_f_enc = tf.random.uniform((batch_size, f_latent_size))
    dummy_z_enc = tf.random.uniform((batch_size, timesteps, z_enc_dim))
    dummy_ext_input = tf.random.uniform((batch_size, timesteps, ext_input_dim))
    gen_s, con_s, co_mean, co_logvar, co_out, fac_s = generator_model([dummy_f_enc, dummy_z_enc, dummy_ext_input])

    import numpy as np
    assert np.array_equal(gen_s.shape.as_list(), [batch_size, timesteps, gen_n_hidden])
    assert np.array_equal(fac_s.shape.as_list(), [batch_size, timesteps, n_factors])

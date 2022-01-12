import tensorflow.keras.layers as tfkl


def create_generator_lfads(params):
    """
    units_gen,
    units_con,
    factors_dim,
    co_dim,
    ext_input_dim,
    inject_ext_input_to_gen,
    """
    from indl.model.lfads.complex import ComplexCell

    # TODO: Sample/Mean from $q(f)$. This will replace the first element in generator init_states
    #  TODO: need a custom function for sample-during-train-mean-during-test. See nn.dropout for inspiration.
    # TODO: Sample from $q(z_t)$, and optionally concat with ext_input, to build generator inputs.

    # TODO: continue generator from lfads-cd/lfadslite.py start at 495
    custom_cell = ComplexCell(
        params['gen_dim'],  # Units in generator GRU
        con_hidden_state_dim,  # Units in controller GRU
        params['factors_dim'],
        params['co_dim'],
        params['ext_input_dim'],
        True,
    )
    generator = tfkl.RNN(custom_cell, return_sequences=True,
                         # recurrent_regularizer=tf.keras.regularizers.l2(l=gen_l2_reg),
                         name='gen_rnn')
    init_states = generator.get_initial_state(gen_input)

    gen_output = generator(gen_input, initial_state=init_states)
    factors = gen_output[-1]
    return factors
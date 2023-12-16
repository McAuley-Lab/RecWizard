Generator Module
****************

.. autoclass:: recwizard.modules.redial.configuration_redial_gen.RedialGenConfig
    :special-members: __init__


.. autoclass:: recwizard.modules.redial.tokenizer_redial_gen.RedialGenTokenizer
    :special-members: __init__
    :members: get_init_kwargs, load_from_dataset, preprocess, collate_fn, encode_plus, batch_encode_plus, process_entities, _fill_movie_occurrences, encodes, tokenize, vocab_size, decode


.. autoclass:: recwizard.modules.redial.modeling_redial_gen.RedialGen
    :special-members: __init__
    :members: forward, response, prepare_input_for_decoder
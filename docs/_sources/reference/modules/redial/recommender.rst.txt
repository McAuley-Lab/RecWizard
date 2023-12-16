Recommender Module
******************

.. autoclass:: recwizard.modules.redial.configuration_redial_rec.RedialRecConfig
    :special-members: __init__

.. autoclass:: recwizard.modules.redial.tokenizer_redial_rec.RedialRecTokenizer
    :special-members: __init__
    :members: get_init_kwargs, load_from_dataset, preprocess, collate_fn, encode_plus, batch_encode_plus, process_entities, _fill_movie_occurrences, encodes, decode

.. autoclass:: recwizard.modules.redial.modeling_redial_rec.RedialRec
    :special-members: __init__
    :members: forward, response

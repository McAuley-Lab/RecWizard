Recommender Module
******************

.. autoclass:: recwizard.modules.kgsf.configuration_kgsf_rec.KGSFRecConfig

.. autoclass:: recwizard.modules.kgsf.tokenizer_kgsf_rec.KGSFRecTokenizer
    :special-members: __init__
    :members: get_init_kwargs, padding_w2v, padding_context, _names_to_id, detect_movie, encode, decode

.. autoclass:: recwizard.modules.kgsf.modeling_kgsf_rec.KGSFRec
    :special-members: __init__
    :members: infomax_loss, get_total_loss, forward, response
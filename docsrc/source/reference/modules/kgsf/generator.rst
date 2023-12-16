Generator Module
****************

.. autoclass:: recwizard.modules.kgsf.configuration_kgsf_gen.KGSFGenConfig

.. autoclass:: recwizard.modules.kgsf.tokenizer_kgsf_gen.KGSFGenTokenizer
    :special-members: __init__
    :members: get_init_kwargs, padding_w2v, padding_context, _names_to_id, detect_movie, encode, decode

.. autoclass:: recwizard.modules.kgsf.modeling_kgsf_gen.KGSFGen
    :special-members: __init__
    :members: _starts, decode_greedy, decode_forced, compute_loss, forward, response
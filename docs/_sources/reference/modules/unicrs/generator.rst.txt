Generator Module
****************

.. autoclass:: recwizard.modules.unicrs.configuration_unicrs_gen.UnicrsGenConfig
    :special-members: __init__

.. autoclass:: recwizard.modules.unicrs.tokenizer_unicrs_gen.UnicrsGenTokenizer
    :special-members: __init__, __call__
    :members: load_from_dataset, mergeEncoding, encodes

.. autoclass:: recwizard.modules.unicrs.modeling_unicrs_gen.UnicrsGen
    :special-members: __init__
    :members: forward, generate, response
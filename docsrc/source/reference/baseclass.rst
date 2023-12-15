BaseClass
---------

recwizard.BaseConfig
^^^^^^^^^^^^^^^^^^^^

The :class:`~BaseConfig` class adds a few features to the Huggingface PretrainedConfig class.

.. autoclass:: recwizard.configuration_utils.BaseConfig
    :special-members: __init__


recwizard.BaseModule
^^^^^^^^^^^^^^^^^^^^

The :class:`~BaseModule` class adds a few features to the Huggingface PreTrainedModel class.

.. autoclass:: recwizard.module_utils.BaseModule
    :special-members: __init__
    :members: get_tokenizer, prepare_weight, response, forward, state_dict, load_state_dict, map_parameters, remove_ignores, load_checkpoint, from_pretrained,


recwizard.BasePipeline
^^^^^^^^^^^^^^^^^^^^^^

The :class:`~BasePipeline` class adds a few features to the Huggingface PreTrainedModel class.

.. autoclass:: recwizard.model_utils.BasePipeline
    :special-members: __init__
    :members: response, forward


recwizard.BaseTokenizer
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~BaseTokenizer` class adds a few features to the Huggingface PreTrainedTokenizer class.

.. autoclass:: recwizard.tokenizer_utils.BaseTokenizer
    :special-members: __init__
    :members: load_from_dataset, unk_token, unk_token_id, vocab_size, _convert_token_to_id, _convert_id_to_token, mergeEncoding, replace_special_tokens, replace_sep_token, replace_bos_token, replace_eos_token, encodes, preprocess, batch_encode_plus, encode_plus, process_entities, decode, get_init_kwargs, save_vocabulary, from_pretrained

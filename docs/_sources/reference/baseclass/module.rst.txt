recwizard.BaseModule
^^^^^^^^^^^^^^^^^^^^

The :class:`~BaseModule` class adds a few features to the Huggingface PreTrainedModel class.

.. autoclass:: recwizard.module_utils.BaseModule
    :special-members: __init__
    :members: get_tokenizer, prepare_weight, response, forward, state_dict, load_state_dict, map_parameters, remove_ignores, load_checkpoint, from_pretrained,

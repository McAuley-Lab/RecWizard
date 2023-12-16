UniCRS
^^^^^^

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


Recommender Module
******************

.. autoclass:: recwizard.modules.unicrs.configuration_unicrs_rec.UnicrsRecConfig
    :special-members: __init__


.. autoclass:: recwizard.modules.unicrs.tokenizer_unicrs_rec.UnicrsRecTokenizer
    :special-members: __init__, __call__
    :members: load_from_dataset, mergeEncoding, encodes, decode


Supporting Modules
******************

.. autoclass:: recwizard.modules.unicrs.kg_prompt.KGPrompt
    :special-members: __init__
    :members: set_and_fix_node_embed, get_entity_embeds, forward


.. autoclass:: recwizard.modules.unicrs.prompt_gpt2.GPT2Attention
    :special-members: __init__
    :members: prune_heads, _split_heads, _merge_heads, _attn, forward

.. autoclass:: recwizard.modules.unicrs.prompt_gpt2.GPT2Block
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.unicrs.prompt_gpt2.GPT2Model
    :special-members: __init__
    :members: parallelize, deparallelize, get_input_embeddings, set_input_embeddings, _prune_heads, forward

.. autoclass:: recwizard.modules.unicrs.prompt_gpt2.PromptGPT2LMHead
    :special-members: __init__
    :members: parallelize, deparallelize, get_output_embeddings, set_output_embeddings, prepare_inputs_for_generation, forward, _reorder_cache

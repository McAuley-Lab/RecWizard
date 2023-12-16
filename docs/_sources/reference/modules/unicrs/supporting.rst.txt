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

Supporting Modules
******************

.. autofunction:: recwizard.modules.kgsf.graph_utils.kaiming_reset_parameters

.. autoclass:: recwizard.modules.kgsf.graph_utils.GraphConvolution
    :special-members: __init__, __repr__
    :members: reset_parameters, forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.GCN
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.GraphAttentionLayer
    :special-members: __init__, __repr__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SelfAttentionLayer
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SelfAttentionLayer_batch
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SelfAttentionLayer2
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.BiAttention
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.GAT
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SpecialSpmmFunction
    :members: forward, backward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SpecialSpmm
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SpGraphAttentionLayer
    :special-members: __init__, __repr__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.graph_utils.SpGAT
    :special-members: __init__
    :members: forward

.. autofunction:: recwizard.modules.kgsf.graph_utils._add_neighbors

.. autofunction:: recwizard.modules.kgsf.transformer_utils._normalize
.. autofunction:: recwizard.modules.kgsf.transformer_utils._build_encoder
.. autofunction:: recwizard.modules.kgsf.transformer_utils._build_encoder4kg
.. autofunction:: recwizard.modules.kgsf.transformer_utils._build_encoder_mask
.. autofunction:: recwizard.modules.kgsf.transformer_utils._build_decoder
.. autofunction:: recwizard.modules.kgsf.transformer_utils._build_decoder4kg
.. autofunction:: recwizard.modules.kgsf.transformer_utils.create_position_codes

.. autoclass:: recwizard.modules.kgsf.transformer_utils.BasicAttention
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.MultiHeadAttention
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerFFN
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerResponseWrapper
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerEncoder4kg
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerEncoderLayer
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerEncoder
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerEncoder_mask
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerDecoderLayer
    :special-members: __init__
    :members: forward, _create_selfattn_mask

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerDecoder
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerDecoderLayerKG
    :special-members: __init__
    :members: forward, _create_selfattn_mask

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerDecoder
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TransformerMemNetModel
    :special-members: __init__
    :members: encode_cand, encode_context_memory, forward

.. autoclass:: recwizard.modules.kgsf.transformer_utils.TorchGeneratorModel
    :special-members: __init__
    :members: _starts, decode_greedy, decode_forced, reorder_encoder_states, reorder_decoder_incremental_state, forward

.. autofunction:: recwizard.modules.kgsf.utils.neginf
.. autofunction:: recwizard.modules.kgsf.utils._create_embeddings
.. autofunction:: recwizard.modules.kgsf.utils._create_entity_embeddings
.. autofunction:: recwizard.modules.kgsf.utils._edge_list
.. autofunction:: recwizard.modules.kgsf.utils._concept_edge_list4GCN
.. autofunction:: recwizard.modules.kgsf.utils.seed_everything
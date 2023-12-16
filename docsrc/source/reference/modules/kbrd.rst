KBRD
^^^^

Generator Module
****************

.. autoclass:: recwizard.modules.kbrd.configuration_kbrd_gen.KBRDGenConfig

.. autoclass:: recwizard.modules.kbrd.tokenizer_kbrd_gen.KBRDGenTokenizer

.. autoclass:: recwizard.modules.kbrd.modeling_kbrd_gen.KBRDGen


Recommender Module
******************

.. autoclass:: recwizard.modules.kbrd.configuration_kbrd_rec.KBRDRecConfig

.. autoclass:: recwizard.modules.kbrd.tokenizer_kbrd_rec.KBRDRecTokenizer

.. autoclass:: recwizard.modules.kbrd.modeling_kbrd_rec.KBRDRec


Supporting Modules
******************

.. autoclass:: recwizard.modules.kbrd.tokenizer_nltk.NLTKTokenizer

.. autofunction:: recwizard.modules.kbrd.tokenizer_nltk.get_tokenizer

.. autofunction:: recwizard.modules.kbrd.tokenizer_nltk.KBRDWordTokenizer

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TorchGeneratorModel
    :special-members: __init__
    :members: _starts, decode_greedy, decode_forced, reorder_encoder_states, reorder_decoder_incremental_state, forward

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TransformerEncoder
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TransformerEncoderLayer
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TransformerDecoder
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TransformerDecoderLayer
    :special-members: __init__
    :members: forward, _create_selfattn_mask

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TransformerGeneratorModel
    :special-members: __init__
    :members: reorder_encoder_states, output

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.BasicAttention
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.MultiHeadAttention
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.kbrd.transformer_encoder_decoder.TransformerFFN
    :special-members: __init__
    :members: forward
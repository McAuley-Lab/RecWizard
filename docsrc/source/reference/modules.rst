Modules
-------

kbrd
~~~~

.. autoclass:: recwizard.modules.kbrd.configuration_kbrd_gen.KBRDGenConfig

.. autoclass:: recwizard.modules.kbrd.configuration_kbrd_rec.KBRDRecConfig

.. autoclass:: recwizard.modules.kbrd.modeling_kbrd_gen.KBRDGen

.. autoclass:: recwizard.modules.kbrd.modeling_kbrd_rec.KBRDRec

.. autoclass:: recwizard.modules.kbrd.tokenizer_kbrd_gen.KBRDGenTokenizer

.. autoclass:: recwizard.modules.kbrd.tokenizer_kbrd_rec.KBRDRecTokenizer




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

kgsf
~~~~

.. autoclass:: recwizard.modules.kgsf.configuration_kgsf_gen.KGSFGenConfig

.. autoclass:: recwizard.modules.kgsf.configuration_kgsf_rec.KGSFRecConfig




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




.. autoclass:: recwizard.modules.kgsf.modeling_kgsf_gen.KGSFGen
    :special-members: __init__
    :members: _starts, decode_greedy, decode_forced, compute_loss, forward, response

.. autoclass:: recwizard.modules.kgsf.modeling_kgsf_rec.KGSFRec
    :special-members: __init__
    :members: infomax_loss, get_total_loss, forward, response




.. autoclass:: recwizard.modules.kgsf.tokenizer_kgsf_gen.KGSFGenTokenizer
    :special-members: __init__
    :members: get_init_kwargs, padding_w2v, padding_context, _names_to_id, detect_movie, encode, decode




.. autoclass:: recwizard.modules.kgsf.tokenizer_kgsf_rec.KGSFRecTokenizer
    :special-members: __init__
    :members: get_init_kwargs, padding_w2v, padding_context, _names_to_id, detect_movie, encode, decode




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




llm
~~~
.. autoclass:: recwizard.modules.llm.configuration_llm.LLMConfig
    :special-members: __init__
.. autoclass:: recwizard.modules.llm.configuration_llm_rec.LLMRecConfig
    :special-members: __init__


.. autoclass:: recwizard.modules.llm.modeling_chatgpt_gen.ChatgptGen
    :special-members: __init__
    :members: from_pretrained, save_pretrained, get_tokenizer, response

.. autoclass:: recwizard.modules.llm.modeling_chatgpt_rec.ChatgptRec
    :special-members: __init__
    :members: from_pretrained, save_pretrained, get_tokenizer, response

.. autoclass:: recwizard.modules.llm.modeling_llama_gen.LlamaGen
    :special-members: __init__
    :members: from_pretrained, save_pretrained, get_tokenizer, response

.. autoclass:: recwizard.modules.llm.tokenizer_chatgpt.ChatgptTokenizer
    :special-members: __init__, __call__


.. autoclass:: recwizard.modules.llm.tokenizer_llama.LlamaTokenizer
    :special-members: __init__
    :members: preprocess

monitor
~~~~~~~

.. autofunction:: recwizard.modules.monitor.monitor.monitoring

.. autoclass:: recwizard.modules.monitor.monitor.RecwizardMonitor
    :special-members: __init__
    :members: monitor


redial
~~~~~~

.. autoclass:: recwizard.modules.redial.autorec.AutoRec
    :special-members: __init__
    :members: load_checkpoint, forward

.. autoclass:: recwizard.modules.redial.autorec.UserEncoder
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.redial.autorec.ReconstructionLoss
    :special-members: __init__
    :members: forward, normalize_loss_reset


.. autofunction:: recwizard.modules.redial.beam_search.get_best_beam
.. autofunction:: recwizard.modules.redial.beam_search.n_gram_repeats
.. autoclass:: recwizard.modules.redial.beam_search.Beam
    :special-members: __init__, __str__
    :members: get_updated_beam, normalized_score

.. autoclass:: recwizard.modules.redial.beam_search.BeamSearch
    :members: initial_beams, update_beams




.. autoclass:: recwizard.modules.redial.configuration_redial_gen.RedialGenConfig
    :special-members: __init__

.. autoclass:: recwizard.modules.redial.configuration_redial_rec.RedialRecConfig
    :special-members: __init__




.. autoclass:: recwizard.modules.redial.hrnn.HRNN
    :special-members: __init__
    :members: get_sentence_representations, forward




.. autoclass:: recwizard.modules.redial.hrnn_for_classification.HRNNForClassification
    :special-members: __init__
    :members: on_pretrain_finished, forward

.. autoclass:: recwizard.modules.redial.hrnn_for_classification.RedialSentimentAnalysisLoss
    :special-members: __init__
    :members: forward




.. autoclass:: recwizard.modules.redial.modeling_redial_gen.RedialGen
    :special-members: __init__
    :members: forward, response, prepare_input_for_decoder
.. autoclass:: recwizard.modules.redial.modeling_redial_gen.DecoderGRU
    :special-members: __init__
    :members: set_pretrained_embeddings, forward
.. autoclass:: recwizard.modules.redial.modeling_redial_gen.SwitchingDecoder
    :special-members: __init__
    :members: set_pretrained_embeddings, forward, replace_movie_with_words, generate




.. autoclass:: recwizard.modules.redial.modeling_redial_rec.RedialRec
    :special-members: __init__
    :members: forward, response


.. autoclass:: recwizard.modules.redial.tokenizer_redial_gen.RedialGenTokenizer
    :special-members: __init__
    :members: get_init_kwargs, load_from_dataset, preprocess, collate_fn, encode_plus, batch_encode_plus, process_entities, _fill_movie_occurrences, encodes, tokenize, vocab_size, decode



.. autoclass:: recwizard.modules.redial.tokenizer_redial_rec.RedialRecTokenizer
    :special-members: __init__
    :members: get_init_kwargs, load_from_dataset, preprocess, collate_fn, encode_plus, batch_encode_plus, process_entities, _fill_movie_occurrences, encodes, decode




.. autoclass:: recwizard.modules.redial.tokenizer_rnn.NLTKTokenizer
    :special-members: __init__
    :members: word_tokenize, nltk_split, pre_tokenize

.. autofunction:: recwizard.modules.redial.tokenizer_rnn.get_tokenizer
.. autofunction:: recwizard.modules.redial.tokenizer_rnn.RnnTokenizer

unicrs
~~~~~~


.. autoclass:: recwizard.modules.unicrs.configuration_unicrs_gen.UnicrsGenConfig
    :special-members: __init__

.. autoclass:: recwizard.modules.unicrs.configuration_unicrs_rec.UnicrsRecConfig
    :special-members: __init__


.. autoclass:: recwizard.modules.unicrs.kg_prompt.KGPrompt
    :special-members: __init__
    :members: set_and_fix_node_embed, get_entity_embeds, forward



.. autoclass:: recwizard.modules.unicrs.modeling_unicrs_gen.UnicrsGen
    :special-members: __init__
    :members: forward, generate, response




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





.. autoclass:: recwizard.modules.unicrs.tokenizer_unicrs_gen.UnicrsGenTokenizer
    :special-members: __init__, __call__
    :members: load_from_dataset, mergeEncoding, encodes

.. autoclass:: recwizard.modules.unicrs.tokenizer_unicrs_rec.UnicrsRecTokenizer
    :special-members: __init__, __call__
    :members: load_from_dataset, mergeEncoding, encodes, decode

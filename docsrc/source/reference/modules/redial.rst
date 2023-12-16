ReDIAL
^^^^^^

Generator Module
****************

.. autoclass:: recwizard.modules.redial.configuration_redial_gen.RedialGenConfig
    :special-members: __init__


.. autoclass:: recwizard.modules.redial.tokenizer_redial_gen.RedialGenTokenizer
    :special-members: __init__
    :members: get_init_kwargs, load_from_dataset, preprocess, collate_fn, encode_plus, batch_encode_plus, process_entities, _fill_movie_occurrences, encodes, tokenize, vocab_size, decode


.. autoclass:: recwizard.modules.redial.modeling_redial_gen.RedialGen
    :special-members: __init__
    :members: forward, response, prepare_input_for_decoder

Recommender Module
******************

.. autoclass:: recwizard.modules.redial.configuration_redial_rec.RedialRecConfig
    :special-members: __init__

.. autoclass:: recwizard.modules.redial.tokenizer_redial_rec.RedialRecTokenizer
    :special-members: __init__
    :members: get_init_kwargs, load_from_dataset, preprocess, collate_fn, encode_plus, batch_encode_plus, process_entities, _fill_movie_occurrences, encodes, decode

.. autoclass:: recwizard.modules.redial.modeling_redial_rec.RedialRec
    :special-members: __init__
    :members: forward, response

Supporting Modules
******************

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


.. autoclass:: recwizard.modules.redial.hrnn.HRNN
    :special-members: __init__
    :members: get_sentence_representations, forward

.. autoclass:: recwizard.modules.redial.hrnn_for_classification.HRNNForClassification
    :special-members: __init__
    :members: on_pretrain_finished, forward

.. autoclass:: recwizard.modules.redial.hrnn_for_classification.RedialSentimentAnalysisLoss
    :special-members: __init__
    :members: forward

.. autoclass:: recwizard.modules.redial.modeling_redial_gen.DecoderGRU
    :special-members: __init__
    :members: set_pretrained_embeddings, forward
.. autoclass:: recwizard.modules.redial.modeling_redial_gen.SwitchingDecoder
    :special-members: __init__
    :members: set_pretrained_embeddings, forward, replace_movie_with_words, generate


.. autoclass:: recwizard.modules.redial.tokenizer_rnn.NLTKTokenizer
    :special-members: __init__
    :members: word_tokenize, nltk_split, pre_tokenize

.. autofunction:: recwizard.modules.redial.tokenizer_rnn.get_tokenizer
.. autofunction:: recwizard.modules.redial.tokenizer_rnn.RnnTokenizer

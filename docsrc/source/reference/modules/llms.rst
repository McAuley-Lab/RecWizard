LLMs
^^^^

General LLM 
***********

.. autoclass:: recwizard.modules.llm.configuration_llm.LLMConfig
    :special-members: __init__

.. autoclass:: recwizard.modules.llm.configuration_llm_rec.LLMRecConfig
    :special-members: __init__

ChatGPT
*******

.. autoclass:: recwizard.modules.llm.modeling_chatgpt_gen.ChatgptGen
    :special-members: __init__
    :members: from_pretrained, save_pretrained, get_tokenizer, response

.. autoclass:: recwizard.modules.llm.modeling_chatgpt_rec.ChatgptRec
    :special-members: __init__
    :members: from_pretrained, save_pretrained, get_tokenizer, response

.. autoclass:: recwizard.modules.llm.tokenizer_chatgpt.ChatgptTokenizer
    :special-members: __init__, __call__

Llama
*****

.. autoclass:: recwizard.modules.llm.modeling_llama_gen.LlamaGen
    :special-members: __init__
    :members: from_pretrained, save_pretrained, get_tokenizer, response

.. autoclass:: recwizard.modules.llm.tokenizer_llama.LlamaTokenizer
    :special-members: __init__
    :members: preprocess


monitor
^^^^^^^

.. autofunction:: recwizard.modules.monitor.monitor.monitoring

.. autoclass:: recwizard.modules.monitor.monitor.RecwizardMonitor
    :special-members: __init__
    :members: monitor

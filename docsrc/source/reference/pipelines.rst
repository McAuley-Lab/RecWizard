Pipelines
`````````


Expansion
^^^^^^^^^

.. autoclass:: recwizard.pipelines.expansion.configuration_expansion.ExpansionConfig
    :special-members: __init__

.. autoclass:: recwizard.pipelines.expansion.modeling_expansion.ExpansionPipeline
    :special-members: __init__
    :members: response, forward

Fill Blank
^^^^^^^^^^

.. autoclass:: recwizard.pipelines.fill_blank.configuration_fill_blank.FillBlankConfig
    :special-members: __init__

.. autoclass:: recwizard.pipelines.fill_blank.modeling_fill_blank.FillBlankPipeline
    :members: response


Switch Decode
^^^^^^^^^^^^^


.. autoclass:: recwizard.pipelines.switch_decode.configuration_switch_decode.SwitchDecodeConfig
    :special-members: __init__

.. autoclass:: recwizard.pipelines.switch_decode.modeling_switch_decode.SwitchDecodePipeline
    :special-members: __init__
    :members: switch_decode, forward, replace_movie_with_words, response

ChatGPT
^^^^^^^

.. autoclass:: recwizard.pipelines.chatgpt.configuration_chatgpt_agent.ChatgptAgentConfig
    :special-members: __init__

.. autoclass:: recwizard.pipelines.chatgpt.modeling_chatgpt_agent.ChatgptAgent
    :special-members: __init__
    :members: response

Trivial
^^^^^^^

.. autoclass:: recwizard.pipelines.trivial.configuration_trivial.TrivialConfig
    :special-members: __init__

.. autoclass:: recwizard.pipelines.trivial.modeling_trivial.TrivialPipeline
    :special-members: __init__
    :members: forward, response

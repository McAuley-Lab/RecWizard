Overview
--------

Code Structure
##############

.. code-block:: bash

    recwizard
  ├── __init__.py
    ├── configuration_utils.py
    ├── model_utils.py
    ├── module_utils.py
    ├── modules
    │   ├── __init__.py
    │   ├── chatgpt
    │   │   ├── __init__.py
    │   │   ├── configuration_chatgpt_gen.py
    │   │   ├── configuration_chatgpt_rec.py
    │   │   ├── modeling_chatgpt_gen.py
    │   │   ├── modeling_chatgpt_rec.py
    │   │   └── tokenizer_chatgpt.py
    │   ├── kbrd
    │   │   ├── configuration_kbrd_gen.py
    │   │   ├── configuration_kbrd_rec.py
    │   │   ├── modeling_kbrd_rec.py
    │   │   ├── shared_encoder.py
    │   │   └── tokenizer_kbrd_rec.py
    │   ├── kgsf
    │   │   ├── __init__.py
    │   │   ├── configuration_kgsf_gen.py
    │   │   ├── configuration_kgsf_rec.py
    │   │   ├── graph_utils.py
    │   │   ├── modeling_kgsf_gen.py
    │   │   ├── modeling_kgsf_rec.py
    │   │   ├── tokenizer_kgsf_gen.py
    │   │   ├── tokenizer_kgsf_rec.py
    │   │   ├── transformer_utils.py
    │   │   └── utils.py
    │   ├── monitor
    │   │   ├── __init__.py
    │   │   └── monitor.py
    │   ├── redial
    │   │   ├── __init__.py
    │   │   ├── autorec.py
    │   │   ├── beam_search.py
    │   │   ├── configuration_redial_gen.py
    │   │   ├── configuration_redial_rec.py
    │   │   ├── hrnn.py
    │   │   ├── hrnn_for_classification.py
    │   │   ├── modeling_redial_gen.py
    │   │   ├── modeling_redial_rec.py
    │   │   ├── params.py
    │   │   ├── tokenizer_redial_gen.py
    │   │   ├── tokenizer_redial_rec.py
    │   │   └── tokenizer_rnn.py
    │   └── unicrs
    │       ├── __init__.py
    │       ├── configuration_unicrs_gen.py
    │       ├── configuration_unicrs_rec.py
    │       ├── kg_prompt.py
    │       ├── modeling_unicrs_gen.py
    │       ├── modeling_unicrs_rec.py
    │       ├── prompt_gpt2.py
    │       ├── tokenizer_unicrs_gen.py
    │       └── tokenizer_unicrs_rec.py
    ├── pipelines
    │   ├── __init__.py
    │   ├── chatgpt
    │   │   ├── __init__.py
    │   │   ├── configuration_chatgpt_agent.py
    │   │   └── modeling_chatgpt_agent.py
    │   ├── expansion
    │   │   ├── __init__.py
    │   │   ├── configuration_expansion.py
    │   │   └── modeling_expansion.py
    │   ├── fill_blank
    │   │   ├── __init__.py
    │   │   ├── configuration_fill_blank.py
    │   │   └── modeling_fill_blank.py
    │   └── switch_decode
    │       ├── __init__.py
    │       ├── configuration_switch_decode.py
    │       └── modeling_switch_decode.py
    ├── tokenizer_utils.py
    └── utility
        ├── __init__.py
        ├── constants.py
        ├── device_manager.py
        ├── entity_linking.py
        ├── singleton.py
        └── utils.py

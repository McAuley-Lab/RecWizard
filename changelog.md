# Change Log

## Simiplify Tokenizers

- [x] delete `tokenization_utils.py`: we are using several existing tokenizers adapted from huggingface to meet the requirements of most crs modules.

- [x] add `tokenizers` folder: to provide some tokenizers that are frequently used in crs modules:
    - [x] `tokenizer_entity.py`: EntityTokenizer, a tokenizer extracting items from <entity> </entity> tags;
    - [x] `tokenizer_nltk.py`: NLTKTokenizer, a tokenizer using NLTK backend tokenizerm, we specially provide this because there are many legacy models (e.g., ReDIAL, KBRD, KGSF) using NLTK-based tokenizations; this is non-trivial since we need to take care of how to serialize cumstomized pre_tokenizer;
    - [x] `tokenizer_multi.py`: MultiTokenizer: a generalized tokenizer, which is not a "real" tokenizer, but a "super" tokenizer to combine multiple tokenizers together, which is a common case in many crs model (e.g., KGSF, KBRD).

- [x] introduce `apply_chat_template` to tokenizers to make e.g., `BOS_TOKEN`, or other tags bind to the specific tokenizers.

- [ ] Fix the bugs:
    - [ ] Concept_tokenizers in KGSF
    - [ ] UniCRS chat template

## Better Organization

- [x] Move `./utilty/monitor.py` to `./monitor_utils.py`, usually import it with `from recwizard import monitor`.
- [x] `MovieLinker` does not follow the design and actually not portable, we downgrade it to hyperlink for google search instead.
- [x] Delete the `DeviceManager` since it is lagecy code.
- [x] Move constants to `__init__.py`.


## Code Style Standardization

- [ ] logging
- [ ] standards for coding styles
    - [ ] Pipelines:
        - [x] Clean `chatgpt/*.py` by changing the `format` to `f-string`.
        - [x] Clean the attributes in `ExpansionConfig` since none of them are used in pipeline.
        - [x] Change the BaseTokenizer type to PreTrainedTokenizerBase type 
        - [x] Clean the attributes in `FillBlankConfig` since one of them is not used in pipeline.
        - [x] Remove `switch_decode/` since those are not used by any modules and still need improvement later.
    - [ ] Modules:
        - [x] kbrd:
            - [x] Change the original file names with the prefix `original_*`, such as `original_entity_attention_encoder.py` and `original_transformer_encoder_encoder.py`.
            - [x] Add reference link in `original_*` python files.
            - [x] Change the `tokenizer_kbrd_rec.py` with simplified tokenizers.
            - [x] Change the `tokenizer_kbrd_gen.py` with simplified tokenizers.
            - [x] Clean `modeling_kbrd_rec.py` and change the `movie*` to `item*`.
            - [x] Clean `modeling_kbrd_gen.py` and change the `movie*` to `item*`.
        - [ ] kgsf:
            - [x] Change the original file names with the prefix `original_*`, such as `original_gcn.py`, `original_transformer.py`, `original_utils.py`.
            - [x] Add reference link in `original_*` python files.
            - [x] Change the `tokenizer_kgsf_rec.py` with simplified tokenizers.
            - [x] Change the `tokenizer_kgsf_gen.py` with simplified tokenizers.
            - [x] Clean `modeling_kgsf_rec.py` and change the `movie*` to `item*`.
            - [x] Clean `modeling_kgsf_gen.py` and change the `movie*` to `item*`.
            - [ ] Add docstring to `modeling_kgsf_rec.py`.
            - [ ] Add docstring to `modeling_kgsf_gen.py`.
        - [ ] unicrs:
            - [x] Change the original file names with the prefix `original_*`, such as `original_kg_prompt.py`, `original_prompt_gpt2.py`.
            - [x] Change the `tokenizer_unicrs_rec.py` with simplified tokenizers.
            - [x] Change the `tokenizer_unicrs_gen.py` with simplified tokenizers.
            - [x] Clean `modeling_unicrs_rec.py` and change the `movie*` to `item*`.
            - [x] Clean `modeling_unicrs_gen.py` and change the `movie*` to `item*`.
            - [ ] Add docstring to `modeling_unicrs_rec.py`.
            - [ ] Add docstring to `modeling_unicrs_gen.py`.
        - [x] llm -> zero_shot:
            - [x] Change the `llm` folder into `chatgpt` and `llama`, which work as two examples of using llms for rec.
            - [x] Finish the `ChatGPT` folder.
            - [x] Finish all the other Huggingface LLMs.
        - [ ] redial
- [x] variable naming
    - [x] Change the occasional camel case namies (e.g., `genIds`) to the underscore names (e.g., `gen_ids`)
    - [x] Change `loadJsonFileFromDataset` to `load_json_file_from_dataset`.
    - [x] Change `recwizard.utility` to `recwizard.utils`
- [ ] rules for importing modules
    - [ ] how to register KGSF tokenizers for MultiTokenizer
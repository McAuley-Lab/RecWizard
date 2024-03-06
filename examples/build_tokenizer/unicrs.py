import os, json
from transformers import AutoTokenizer
from tokenizers.normalizers import Replace, Sequence

from recwizard.utils import create_chat_message

from recwizard.tokenizers import EntityTokenizer
from recwizard.modules.unicrs.tokenizer_unicrs_rec import UnicrsRecTokenizer
from recwizard.modules.unicrs.tokenizer_unicrs_gen import UnicrsGenTokenizer


TEST_MESSAGE = "User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!"


def build_unicrs_rec_tokenizer(path="../local_repo/unicrs-rec"):
    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Initialize the context (DialoGPT) and prompt (RoBERTa) tokenizers
    context_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", truncation_side="left")
    prompt_tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation_side="left")
    context_tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "additional_special_tokens": ["<movie>"],
        }
    )
    prompt_tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<movie>"],
        }
    )

    context_tokenizer.backend_tokenizer.normalizer = Sequence(
        [Replace("<entity>", ""), Replace("</entity>", ""), Replace("_", " ")]
    )
    prompt_tokenizer.backend_tokenizer.normalizer = Sequence(
        [Replace("<entity>", ""), Replace("</entity>", ""), Replace("_", " ")]
    )

    context_tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'System: ' + message['content'] }}{% endif %}{{ eos_token }}{% endfor %}"
    prompt_tokenizer.chat_template = "{{cls_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'System: ' + message['content'] }}{% endif %}{{ sep_token }}{% endfor %}"

    # Multi tokenizer
    tokenizers = {
        "item": entity_tokenizer,
        "context": context_tokenizer,
        "prompt": prompt_tokenizer,
    }

    tokenizer = UnicrsRecTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="item")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    tokenizer.save_pretrained("test")

    # Evaluate the saved multi tokenizer
    tokenizer = UnicrsRecTokenizer.from_pretrained("test")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Save raw vocab
    os.makedirs(os.path.join("test", "raw_vocab"), exist_ok=True)
    json.dump(entity2id, open(os.path.join("test", "raw_vocab", "entity2id.json"), "w"))

    return tokenizer


def build_unicrs_gen_tokenizer(path="../local_repo/unicrs-gen"):
    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Initialize the context (DialoGPT) and prompt (RoBERTa) tokenizers
    context_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", truncation_side="left")
    prompt_tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation_side="left")
    context_tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "additional_special_tokens": ["<movie>"],
        }
    )
    prompt_tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<movie>"],
        }
    )

    context_tokenizer.backend_tokenizer.normalizer = Sequence(
        [Replace("<entity>", ""), Replace("</entity>", ""), Replace("_", " ")]
    )
    prompt_tokenizer.backend_tokenizer.normalizer = Sequence(
        [Replace("<entity>", ""), Replace("</entity>", ""), Replace("_", " ")]
    )

    context_tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'System: ' + message['content'] }}{% endif %}{{ eos_token }}{% endfor %}{{'System: '}}"
    prompt_tokenizer.chat_template = "{{cls_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'System: ' + message['content'] }}{% endif %}{{ sep_token }}{% endfor %}{{'System: '}}"

    # Multi tokenizer
    tokenizers = {
        "item": entity_tokenizer,
        "context": context_tokenizer,
        "prompt": prompt_tokenizer,
    }

    tokenizer = UnicrsGenTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="context")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    tokenizer.save_pretrained("test")

    # Evaluate the saved multi tokenizer
    tokenizer = UnicrsGenTokenizer.from_pretrained("test")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Save raw vocab
    os.makedirs(os.path.join("test", "raw_vocab"), exist_ok=True)
    json.dump(entity2id, open(os.path.join("test", "raw_vocab", "entity2id.json"), "w"))

    return tokenizer


if __name__ == "__main__":
    rec_tokenizer = build_unicrs_rec_tokenizer()
    gen_tokenizer = build_unicrs_gen_tokenizer()

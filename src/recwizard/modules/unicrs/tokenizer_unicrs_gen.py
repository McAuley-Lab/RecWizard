from recwizard.tokenizers import EntityTokenizer
from recwizard.tokenizers import MultiTokenizer as UnicrsGenTokenizer


if __name__ == "__main__":

    import os, json
    from transformers import AutoTokenizer
    from tokenizers.normalizers import Replace, Sequence

    path = "../../../../../local_repo/unicrs-gen"

    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    # Initialize the context (DialoGPT) and prompt (RoBERTa) tokenizers
    context_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", truncation_side="left")
    prompt_tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation_side="left")

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

    # tokenizers
    tokenizers = {
        "item": entity_tokenizer,
        "context": context_tokenizer,
        "prompt": prompt_tokenizer,
    }

    tokenizer = UnicrsGenTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="context")

    print(tokenizer(f"System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?"))

    tokenizer.save_pretrained(path)

    # Load the new tokenizer
    tokenizer = UnicrsGenTokenizer.from_pretrained(path)
    print(tokenizer(f"System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?"))

    # Save raw vocab
    import os, json

    os.makedirs(os.path.join(path, "raw_vocab"), exist_ok=True)
    json.dump(entity2id, open(os.path.join(path, "raw_vocab", "entity2id.json"), "w"))

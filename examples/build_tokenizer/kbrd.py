from recwizard.utils import create_chat_message

from recwizard.tokenizers import EntityTokenizer, NLTKTokenizer
from recwizard.modules.kbrd.tokenizer_kbrd_rec import KBRDRecTokenizer
from recwizard.modules.kbrd.tokenizer_kbrd_gen import KBRDGenTokenizer

import os, json

TEST_MESSAGE = "User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!"


def build_kbrd_rec_tokenizer(path="../local_repo/kbrd-rec"):
    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    tokenizer = KBRDRecTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Save the tokenizer
    tokenizer.save_pretrained("test")

    # Evaluate the saved tokenizer
    tokenizer = KBRDRecTokenizer.from_pretrained("test")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(tokenizer.apply_chat_template(chat_message, tokenize=True))

    return tokenizer


def build_kbrd_gen_tokenizer(path="../local_repo/kbrd-gen"):

    # Initialize the NLTK tokenizer
    word2id = json.load(open(os.path.join(path, "raw_vocab", "word2id.json")))

    nltk_tokenizer = NLTKTokenizer(
        vocab=word2id,
        nltk_sent_language="english",
        nltk_word_tokenizer="treebank",
        unk_token="__unk__",
        pad_token="__null__",
        bos_token="__start__",
        eos_token="__end__",
    )

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(nltk_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(nltk_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Assemble the MultiTokenizer as a combination of the two tokenizers
    tokenizers = {
        "nltk": nltk_tokenizer,
        "entity": entity_tokenizer,
    }

    tokenizer = KBRDGenTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="nltk")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Save the multi tokenizer
    tokenizer.save_pretrained("test")

    # Evaluate the saved multi tokenizer
    tokenizer = KBRDGenTokenizer.from_pretrained("test")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(tokenizer.apply_chat_template(chat_message, tokenize=True))

    return tokenizer


if __name__ == "__main__":
    rec_tokenizer = build_kbrd_rec_tokenizer()
    gen_tokenizer = build_kbrd_gen_tokenizer()

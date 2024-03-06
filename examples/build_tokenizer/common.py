from recwizard.tokenizers import EntityTokenizer, NLTKTokenizer, MultiTokenizer
from recwizard.utils import create_chat_message

import os, json

TEST_MESSAGE = "User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!"


def build_entity_tokenizer(path="../local_repo/kbrd-gen/raw_vocab/"):

    entity2id = json.load(open(os.path.join(path, "entity2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    entity_tokenizer.save_pretrained("test")
    entity_tokenizer = EntityTokenizer.from_pretrained("test")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    return entity_tokenizer


def build_word_tokenizer(path="../local_repo/kbrd-gen/raw_vocab/"):

    word2id = json.load(open(os.path.join(path, "word2id.json")))
    word_tokenizer = NLTKTokenizer(vocab=word2id, unk_token="[UNK]", pad_token="[PAD]")

    chat = create_chat_message(TEST_MESSAGE)
    print(word_tokenizer.apply_chat_template(chat, tokenize=False))
    print(word_tokenizer.apply_chat_template(chat, tokenize=True))

    word_tokenizer.save_pretrained("test")
    word_tokenizer = NLTKTokenizer.from_pretrained("test")

    chat = create_chat_message(TEST_MESSAGE)
    print(word_tokenizer.apply_chat_template(chat, tokenize=False))
    print(word_tokenizer.apply_chat_template(chat, tokenize=True))

    return word_tokenizer


def build_multi_tokenizer(entity_tokenizer, word_tokenizer):

    tokenizers = {"entity": entity_tokenizer, "word": word_tokenizer}
    multi_tokenizer = MultiTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="entity")

    chat = create_chat_message(TEST_MESSAGE)
    print(multi_tokenizer.apply_chat_template(chat, tokenize=False))
    print(multi_tokenizer.apply_chat_template(chat, tokenize=True))

    multi_tokenizer.save_pretrained("test")
    multi_tokenizer = MultiTokenizer.from_pretrained("test")

    chat = create_chat_message(TEST_MESSAGE)
    print(multi_tokenizer.apply_chat_template(chat, tokenize=False))
    print(multi_tokenizer.apply_chat_template(chat, tokenize=True))

    return multi_tokenizer


if __name__ == "__main__":
    entity_tokenizer = build_entity_tokenizer()
    word_tokenizer = build_word_tokenizer()
    multi_tokenizer = build_multi_tokenizer(entity_tokenizer, word_tokenizer)

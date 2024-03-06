from recwizard.utils import create_chat_message, START_TAG, END_TAG

from recwizard.tokenizers import EntityTokenizer, NLTKTokenizer
from recwizard.modules.redial import RedialGenTokenizer, RedialRecTokenizer

from transformers import AutoTokenizer
from tokenizers.normalizers import Replace, Sequence

import os, json

TEST_MESSAGE = "User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!"


def build_redial_rec_tokenizer(path="../local_repo/redial-rec"):
    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "item2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]")

    # Sentiment tokenizer
    sen_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")
    sen_tokenizer.backend_tokenizer.normalizer = Sequence(
        [Replace(START_TAG, ""), Replace(END_TAG, ""), Replace("_", " ")]
    )

    # Tokenizer for the recommender
    rec_tokenizer = RedialRecTokenizer(
        tokenizers={
            "entity": entity_tokenizer,
            "sen_encoding": sen_tokenizer,
        },
        tokenizer_key_for_decoding="entity",
    )

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(rec_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(rec_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Save the tokenizer
    rec_tokenizer.save_pretrained("test")

    # Evaluate the saved tokenizer
    rec_tokenizer = RedialRecTokenizer.from_pretrained("test")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(rec_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(rec_tokenizer.apply_chat_template(chat_message, tokenize=True))

    return rec_tokenizer


def build_redial_gen_tokenizer(path="../local_repo/redial-gen"):
    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "item2id.json")))
    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]")

    # Sentiment tokenizer
    sen_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")
    sen_tokenizer.backend_tokenizer.normalizer = Sequence(
        [Replace(START_TAG, ""), Replace(END_TAG, ""), Replace("_", " ")]
    )

    # RNN tokenizer
    word2id = json.load(open(os.path.join(path, "raw_vocab", "word2id.json")))
    rnn_tokenizer = NLTKTokenizer(
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        vocab=word2id,
    )

    # Tokenizer for the generator
    gen_tokenizer = RedialGenTokenizer(
        tokenizers={
            "dialogue": rnn_tokenizer,
            "entity": entity_tokenizer,
            "sen_encoding": sen_tokenizer,
        },
        tokenizer_key_for_decoding="dialogue",
    )

    # Evaluate the tokenizer
    chat_message = create_chat_message(TEST_MESSAGE)
    print(gen_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(gen_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Save the tokenizer
    gen_tokenizer.save_pretrained("test")

    # Evaluate the saved tokenizer
    gen_tokenizer = RedialGenTokenizer.from_pretrained("test")

    chat_message = create_chat_message(TEST_MESSAGE)
    print(gen_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(gen_tokenizer.apply_chat_template(chat_message, tokenize=True))

    return gen_tokenizer


if __name__ == "__main__":
    rec_tokenizer = build_redial_rec_tokenizer()
    gen_tokenizer = build_redial_gen_tokenizer()

import os, json

from recwizard.modules.kgsf.tokenizer_kgsf_base import KGSFBaseTokenizer
from recwizard.modules.kgsf.tokenizer_kgsf_rec import KGSFRecTokenizer
from recwizard.modules.kgsf.tokenizer_kgsf_gen import KGSFGenTokenizer

from recwizard.tokenizers import EntityTokenizer
from recwizard.utils import create_chat_message

TEST_MESSAGE = "User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!"


def build_kgsf_rec_tokenizer(path="../local_repo/kgsf-rec"):

    # Step 1. Load the raw vocab
    item2id = json.load(open(os.path.join(path, "raw_vocab", "item2id.json")))
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    concept2id = json.load(open(os.path.join(path, "raw_vocab", "concept2id.json")))

    # Step 2. itemtokenizer: item2id, fill the holes
    item_tokenizer = EntityTokenizer(vocab=item2id, unk_token="[UNK]", pad_token="[PAD]")

    print("item tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(item_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(item_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 3. entitytokenizer: entity2id, fill the holes
    entity_tokenizer = KGSFBaseTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    print("entity tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 4. concepttokenizer: concept2id, fill the holes
    concept_tokenizer = KGSFBaseTokenizer(vocab=concept2id, unk_token="[UNK]", pad_token="[UNK]", lowercase=True)

    print("concept tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(concept_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(concept_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 5. Assemble them into a MultiTokenizer and save it

    tokenizers = {
        "item": item_tokenizer,
        "entity": entity_tokenizer,
        "concept": concept_tokenizer,
    }

    # Step 6. Use the MultiTokenizer and save it
    multi_tokenizer = KGSFRecTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="item")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=True))

    multi_tokenizer.save_pretrained("test")
    multi_tokenizer = KGSFRecTokenizer.from_pretrained("test")
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=True))

    return multi_tokenizer


def build_kgsf_gen_tokenizer(path="../local_repo/kgsf-gen"):

    # Step 1. Load the raw vocab
    item2id = json.load(open(os.path.join(path, "raw_vocab", "item2id.json")))
    word2id = json.load(open(os.path.join(path, "raw_vocab", "word2id.json")))
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    concept2id = json.load(open(os.path.join(path, "raw_vocab", "concept2id.json")))

    # Step 2. wordtokenizer: word2id, fill the holes
    word2id = json.load(open(os.path.join(path, "raw_vocab", "word2id.json")))
    word_tokenizer = KGSFBaseTokenizer(
        vocab=word2id, unk_token="[UNK]", pad_token="[PAD]", start_token="[START]", end_token="[END]"
    )

    print("word tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(word_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(word_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 3. itemtokenizer: item2id, fill the holes
    item_tokenizer = EntityTokenizer(vocab=item2id, unk_token="[UNK]", pad_token="[PAD]")

    print("item tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(item_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(item_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 4. entitytokenizer: entity2id, fill the holes
    entity_tokenizer = KGSFBaseTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    print("entity tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(entity_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 5. concepttokenizer: concept2id, fill the holes
    concept_tokenizer = KGSFBaseTokenizer(vocab=concept2id, unk_token="[UNK]", pad_token="[UNK]", lowercase=True)

    print("concept tokenizer")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(concept_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(concept_tokenizer.apply_chat_template(chat_message, tokenize=True))

    # Step 6. Assemble them into a MultiTokenizer and save it

    tokenizers = {
        "word": word_tokenizer,
        "item": item_tokenizer,
        "entity": entity_tokenizer,
        "concept": concept_tokenizer,
    }

    # Step 7. Use the MultiTokenizer and save it
    multi_tokenizer = KGSFGenTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="word")
    chat_message = create_chat_message(TEST_MESSAGE)
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=True))

    multi_tokenizer.save_pretrained("test")
    multi_tokenizer = KGSFGenTokenizer.from_pretrained("test")
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=False))
    print(multi_tokenizer.apply_chat_template(chat_message, tokenize=True))

    return multi_tokenizer


if __name__ == "__main__":
    build_kgsf_gen_tokenizer()
    build_kgsf_rec_tokenizer()

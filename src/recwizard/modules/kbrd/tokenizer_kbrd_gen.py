from recwizard.tokenizers import EntityTokenizer, NLTKTokenizer
from recwizard.tokenizers import MultiTokenizer as KBRDGenTokenizer


if __name__ == "__main__":
    """How to use EntityTokenizer and NLTKTokenizer to build tokenizers as KBRDGenTokenizer."""

    import os, json

    path = "../../../../../local_repo/kbrd-gen"

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

    print(nltk_tokenizer(f"I like <entity>Avatar</entity>!"))

    # Initialize the Entity tokenizer
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))

    entity_tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")
    print(entity_tokenizer(f"I like <entity>The_Godfather</entity>!"))

    # Assemble the MultiTokenizer as a combination of the two tokenizers
    tokenizers = {
        "nltk": nltk_tokenizer,
        "entity": entity_tokenizer,
    }

    multi_tokenizer = KBRDGenTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="nltk")

    print(multi_tokenizer(f"I like <entity>Avatar</entity>!"))
    multi_tokenizer.save_pretrained(path)

    # Evaluate the saved tokenizer
    multi_tokenizer = KBRDGenTokenizer.from_pretrained(path)
    print(multi_tokenizer(f"I like <entity>Avatar</entity>!"))

    # Evaluate the saved tokenizer
    from recwizard.utility import create_chat_message

    test = create_chat_message("I like <entity>The_Godfather</entity>!")
    print(multi_tokenizer.apply_chat_template(test, tokenize=False))

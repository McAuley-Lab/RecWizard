from recwizard.tokenizers import EntityTokenizer as KBRDRecTokenizer

if __name__ == "__main__":
    """How to use EntityTokenizer to build a tokenizer as KBRDRecTokenizer."""

    import os, json

    # Initialize the Entity tokenizer
    path = "../../../../../local_repo/kbrd-rec"
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    tokenizer = KBRDRecTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")
    print(tokenizer(f"I like <entity>The_Godfather</entity>!"))

    # Save the tokenizer
    tokenizer.save_pretrained(path)

    # Evaluate the saved tokenizer
    tokenizer = KBRDRecTokenizer.from_pretrained(path)
    print(tokenizer(f"I like <entity>The_Godfather</entity>!"))

    # Evaluate the saved tokenizer
    from recwizard.utils import create_chat_message

    test = create_chat_message("I like <entity>The_Godfather</entity>!")
    print(tokenizer.apply_chat_template(test, tokenize=False))

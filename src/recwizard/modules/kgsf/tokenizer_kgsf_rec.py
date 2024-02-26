"""
    MultiTokenizer(
        tokenizers={
            'item': KGSFBaseTokenizer(vocab=item2id),
            'entity': EntityTokenizer(vocab=entity2id),
            'concept': KGSFTokenizer(vocab=concept2id)
        },
        tokenizer_key_for_decoding='item',
    )
"""

from recwizard.modules.kgsf.tokenizer_kgsf_base import KGSFBaseTokenizer

from recwizard.tokenizers import EntityTokenizer
from recwizard.tokenizers import MultiTokenizer as KGSFRecTokenizer


if __name__ == "__main__":
    import os, json

    # Step 1. Load the raw vocab
    path = "../../../../../local_repo/kgsf-rec"
    item2id = json.load(open(os.path.join(path, "raw_vocab", "item2id.json")))
    entity2id = json.load(open(os.path.join(path, "raw_vocab", "entity2id.json")))
    concept2id = json.load(open(os.path.join(path, "raw_vocab", "concept2id.json")))

    string = "She also voiced a character in <entity>The_Seven-Ups</entity> and <entity>fwjeljflwej</entity>"

    # Step 2. itemtokenizer: item2id, fill the holes
    item_tokenizer = EntityTokenizer(vocab=item2id, unk_token="[UNK]", pad_token="[PAD]")

    print(item_tokenizer(string))
    print(item_tokenizer.tokenize(string))

    # Step 3. entitytokenizer: entity2id, fill the holes
    entity_tokenizer = KGSFBaseTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    print(entity_tokenizer(string))
    print(entity_tokenizer.tokenize(string))

    # Step 4. concepttokenizer: concept2id, fill the holes
    concept_tokenizer = KGSFBaseTokenizer(vocab=concept2id, unk_token="[UNK]", pad_token="[UNK]", lowercase=True)

    print(concept_tokenizer(string))
    print(concept_tokenizer.tokenize(string))

    # Step 5. Assemble them into a MultiTokenizer and save it

    tokenizers = {
        "item": item_tokenizer,
        "entity": entity_tokenizer,
        "concept": concept_tokenizer,
    }

    # Step 6. Use the MultiTokenizer and save it
    multi_tokenizer = KGSFRecTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="item")
    print(multi_tokenizer(string))

    multi_tokenizer.save_pretrained(path)
    multi_tokenizer = KGSFRecTokenizer.from_pretrained(path)
    print(multi_tokenizer(string))

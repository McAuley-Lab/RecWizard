"""
    MultiTokenizer(
        tokenizers={
            'word': KGSFBaseTokenizer(vocab=word2id),
            'item': KGSFBaseTokenizer(vocab=item2id),
            'entity': EntityTokenizer(vocab=entity2id),
            'concept': EntityTokenizer(vocab=concept2id)
        },
        tokenizer_key_for_decoding='word',
    )
"""

from recwizard.modules.kgsf.tokenizer_kgsf_base import KGSFBaseTokenizer

from recwizard.tokenizers import EntityTokenizer
from recwizard.tokenizers import MultiTokenizer as KGSFGenTokenizer

if __name__ == "__main__":
    import os, json

    DIR = "../../../../../local_repo/kgsf-gen"
    string = (
        "She also voiced a character in <entity>The_Seven-Ups</entity> and <entity>Angels_with_Dirty_Faces</entity>"
    )

    # Step 1. wordtokenizer: word2id, fill the holes
    word2id = json.load(open(os.path.join(DIR, "raw_vocab", "word2id.json")))
    word_tokenizer = KGSFBaseTokenizer(
        vocab=word2id, unk_token="[UNK]", pad_token="[PAD]", start_token="[START]", end_token="[END]"
    )
    word_tokenizer.add_special_tokens({"additional_special_tokens": ["[START]", "[END]"]})

    print(word_tokenizer(string))
    print(word_tokenizer.tokenize(string))

    # Step 2. itemtokenizer: item2id, fill the holes
    item2id = json.load(open(os.path.join(DIR, "raw_vocab", "item2id.json")))
    item_tokenizer = EntityTokenizer(vocab=item2id, unk_token="[UNK]", pad_token="[PAD]")

    print(item_tokenizer(string))
    print(item_tokenizer.tokenize(string))

    # Step 3. entitytokenizer: entity2id, fill the holes
    entity2id = json.load(open(os.path.join(DIR, "raw_vocab", "entity2id.json")))
    entity_tokenizer = KGSFBaseTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    print(entity_tokenizer(string))
    print(entity_tokenizer.tokenize(string))

    # Step 4. concepttokenizer: concept2id, fill the holes
    concept2id = json.load(open(os.path.join(DIR, "raw_vocab", "concept2id.json")))
    concept_tokenizer = KGSFBaseTokenizer(vocab=concept2id, unk_token="[UNK]", pad_token="[UNK]", lowercase=True)

    print(concept_tokenizer(string))
    print(concept_tokenizer.tokenize(string))

    # Step 5. Assemble them into a MultiTokenizer and save it

    tokenizers = {
        "word": word_tokenizer,
        "item": item_tokenizer,
        "entity": entity_tokenizer,
        "concept": concept_tokenizer,
    }

    multi_tokenizer = KGSFGenTokenizer(tokenizers=tokenizers, tokenizer_key_for_decoding="word")
    print(multi_tokenizer(string))

    # Step 6. Load the MultiTokenizer and test it
    multi_tokenizer.save_pretrained(DIR, push_to_hub=False)
    multi_tokenizer = KGSFGenTokenizer.from_pretrained(DIR)
    print(multi_tokenizer(string))

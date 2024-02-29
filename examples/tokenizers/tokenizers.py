from recwizard.tokenizers import EntityTokenizer, NLTKTokenizer, MultiTokenizer
from recwizard.utils import create_chat_message

import os, json

DIR = "../local_repo/kbrd-gen/raw_vocab/"

# EntityTokenizer
entity2id = json.load(open(os.path.join(DIR, "entity2id.json")))
tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

chat_message = create_chat_message("User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!")
print(tokenizer.apply_chat_template(chat_message, tokenize=False))
print(tokenizer.apply_chat_template(chat_message, tokenize=True))

tokenizer.save_pretrained("test")
tokenizer = EntityTokenizer.from_pretrained("test")

chat_message = create_chat_message("User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!")
print(tokenizer.apply_chat_template(chat_message, tokenize=False))
print(tokenizer.apply_chat_template(chat_message, tokenize=True))

# TODO: add NLTKTokenizer and MultiTokenizer

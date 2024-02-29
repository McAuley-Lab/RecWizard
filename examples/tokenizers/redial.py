from recwizard.modules.redial import (
    RedialGen,
    RedialRec,
    RedialGenTokenizer,
    RedialRecTokenizer,
)
from recwizard.utils import load_json_file_from_dataset, START_TAG, END_TAG
from recwizard.tokenizers import NLTKTokenizer, EntityTokenizer

from transformers import AutoTokenizer
from tokenizers.normalizers import Replace, Sequence

# Load related vocabs
dataset = "redial"
vocab = load_json_file_from_dataset(dataset, "vocab.json")
id2entity = load_json_file_from_dataset(dataset, "id2entity.json")

# Build word2id and item2id
word2id = {word: i for i, word in enumerate(vocab)}
item2id = {f"{START_TAG}{item.replace(' ', '_')}{END_TAG}": int(i) for i, item in id2entity.items()}

# Fill the holes of item2id to make indices consecutive
valid_indices = set(item2id.values())
for i in range(max(valid_indices) + 1):
    if i not in valid_indices:
        item2id[f"[FILL{i}]"] = i
item2id["[UNK]"] = max(valid_indices) + 1
item2id["[PAD]"] = max(valid_indices) + 2

# Sentiment tokenizer
sen_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")
sen_tokenizer.backend_tokenizer.normalizer = Sequence([Replace(START_TAG, ""), Replace(END_TAG, ""), Replace("_", " ")])

# RNN tokenizer
rnn_tokenizer = NLTKTokenizer(
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
    vocab=word2id,
)

query = (
    "User: Hi I am looking for a movie like <entity>Super_Troopers_(2001)</entity>"
    "<sep>System: You should watch <entity>Police_Academy_(1984)</entity>"
    "<sep>User: Is that a great one? I have never seen it. I have seen <entity>American_Pie</entity>"
)

# Tokenizer for the generator
gen_tokenizer = RedialGenTokenizer(
    tokenizers={
        "dialogue": rnn_tokenizer,
        "entity": EntityTokenizer(vocab=item2id, unk_token="[UNK]"),
        "sen_encoding": sen_tokenizer,
    },
    tokenizer_key_for_decoding="dialogue",
)

gen_module = RedialGen.from_pretrained("recwizard/redial-gen")
result = gen_module.response(query, gen_tokenizer)
print(result)

# Tokenizer for the recommender
rec_tokenizer = RedialRecTokenizer(
    tokenizers={
        "entity": EntityTokenizer(vocab=item2id, unk_token="[UNK]"),
        "sen_encoding": sen_tokenizer,
    },
    tokenizer_key_for_decoding="entity",
)

rec_module = RedialRec.from_pretrained("recwizard/redial-rec")
result = rec_module.response(query, rec_tokenizer)

# Save the tokenizers and modules
gen_path = "../local_repo/redial-gen"
rec_path = "../local_repo/redial-rec"

gen_tokenizer.save_pretrained(gen_path)
rec_tokenizer.save_pretrained(rec_path)

gen_module.save_pretrained(gen_path)
rec_module.save_pretrained(rec_path)

# Raw data
import os, json

os.makedirs(os.path.join(gen_path, "raw_vocab"), exist_ok=True)
os.makedirs(os.path.join(rec_path, "raw_vocab"), exist_ok=True)

with open(os.path.join(gen_path, "raw_vocab", "word2id.json"), "w") as f:
    json.dump(word2id, f)

with open(os.path.join(gen_path, "raw_vocab", "item2id.json"), "w") as f:
    json.dump(item2id, f)

with open(os.path.join(rec_path, "raw_vocab", "item2id.json"), "w") as f:
    json.dump(item2id, f)

# Test the saved model
gen_module = RedialGen.from_pretrained(gen_path)
rec_module = RedialRec.from_pretrained(rec_path)

gen_tokenizer = RedialGenTokenizer.from_pretrained(gen_path)
rec_tokenizer = RedialRecTokenizer.from_pretrained(rec_path)

print(gen_module.response(query, gen_tokenizer))
print(rec_module.response(query, rec_tokenizer))

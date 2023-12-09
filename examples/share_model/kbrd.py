import os
import sys

sys.path.append("./src/recwizard")

from modules.kbrd import KBRDGen, KBRDRec, KBRDGenTokenizer, KBRDRecTokenizer
from pipelines.trivial import TrivialConfig, TrivialPipeline
from utility import HF_ORG

dir = "path/to/load"  # any local dir

# load rec
rec_tokenizer = KBRDRecTokenizer.from_pretrained(os.path.join(dir, "temp_kbrd_rec"))
rec_module = KBRDRec.from_pretrained(os.path.join(dir, "temp_kbrd_rec"))

## load gen
gen_tokenizer = KBRDGenTokenizer.from_pretrained(os.path.join(dir, "temp_kbrd_gen"))
gen_module = KBRDGen.from_pretrained(os.path.join(dir, "temp_kbrd_gen"))

# load tirival pipeline
pipeline = TrivialPipeline(
    config=TrivialConfig(),
    rec_module=rec_module,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    gen_tokenizer=gen_tokenizer,
)

print(pipeline.response("I like <entity>Avatar</entity>, and you?", return_dict=False))

# result:

# __start__ i'm good. what kind of movies do you like? __end__
#  - Avatar
#  - Trojan War
#  - The Patriot

# push to hub

gen_repo = os.path.join(HF_ORG, f"kbrd-gen-redial")
gen_module.push_to_hub(gen_repo)
gen_tokenizer.push_to_hub(gen_repo)

rec_repo = os.path.join(HF_ORG, f"kbrd-rec-redial")
rec_module.push_to_hub(rec_repo)
rec_tokenizer.push_to_hub(rec_repo)

# test after pushed to hub

# load rec
rec_tokenizer = KBRDRecTokenizer.from_pretrained(rec_repo)
rec_module = KBRDRec.from_pretrained(rec_repo)

## load gen
gen_tokenizer = KBRDGenTokenizer.from_pretrained(gen_repo)
gen_module = KBRDGen.from_pretrained(gen_repo)

# load tirival pipeline
pipeline = TrivialPipeline(
    config=TrivialConfig(),
    rec_module=rec_module,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    gen_tokenizer=gen_tokenizer,
)

print(pipeline.response("I like <entity>Avatar</entity>, and you?", return_dict=False))

# __start__ i'm good. what kind of movies do you like? __end__
#  - Avatar
#  - Trojan War
#  - The Patriot
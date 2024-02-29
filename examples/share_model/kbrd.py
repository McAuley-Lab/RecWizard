import os
import sys

sys.path.append("./src/recwizard")

from recwizard.modules.kbrd import KBRDGen, KBRDRec, KBRDGenTokenizer, KBRDRecTokenizer
from recwizard.pipelines.trivial import TrivialConfig, TrivialPipeline
from recwizard.utils import HF_ORG

dir = "path/to/load"  # any local dir

# load rec
rec_tokenizer = KBRDRecTokenizer.from_pretrained(os.path.join(dir, "kbrd-rec"))
rec_module = KBRDRec.from_pretrained(os.path.join(dir, "kbrd-rec"))

## load gen
gen_tokenizer = KBRDGenTokenizer.from_pretrained(os.path.join(dir, "kbrd-gen"))
gen_module = KBRDGen.from_pretrained(os.path.join(dir, "kbrd-gen"))

# load tirival pipeline
pipeline = TrivialPipeline(
    config=TrivialConfig(),
    rec_module=rec_module,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    gen_tokenizer=gen_tokenizer,
)

print(
    pipeline.response(
        query="User: Hello; <sep> System: Hi <sep> User: Could you recommend me some movies similar to <entity>Avatar</entity>?",
        return_dict=False,
    )
)
# push to hub

gen_repo = os.path.join(HF_ORG, f"kbrd-gen")
gen_module.push_to_hub(gen_repo)
gen_tokenizer.push_to_hub(gen_repo)

rec_repo = os.path.join(HF_ORG, f"kbrd-rec")
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

print(
    pipeline.response(
        "User: Hello; <sep> System: Hi <sep> User: Could you recommend me some movies similar to <entity>Avatar</entity>?",
        return_dict=False,
    )
)

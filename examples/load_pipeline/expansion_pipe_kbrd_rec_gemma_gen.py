from transformers import AutoTokenizer
from recwizard.modules.kbrd.modeling_kbrd_rec import KBRDRec, KBRDRecTokenizer
from recwizard.modules.zero_shot.modeling_hf_llm_gen import HFLLMGenConfig, HFLLMGen

from recwizard.pipelines.expansion import ExpansionPipeline, ExpansionConfig

gen_repo = "google/gemma-2b-it"
rec_repo = "../local_repo/kbrd-rec"

# test after pushed to hub

# load rec
rec_tokenizer = KBRDRecTokenizer.from_pretrained(rec_repo)
rec_module = KBRDRec.from_pretrained(rec_repo)
rec_module.to("cuda")

## load gen
gen_tokenizer = AutoTokenizer.from_pretrained(gen_repo)
gen_module = HFLLMGen(HFLLMGenConfig(model_name=gen_repo))
gen_module.to("cuda")

# load expansion pipeline
pipeline = ExpansionPipeline(
    config=ExpansionConfig(),
    rec_module=rec_module,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    gen_tokenizer=gen_tokenizer,
)

print(
    pipeline.response(
        query="User: Hello; <sep> System: Hi <sep> User: Could you recommend me some movies similar to <entity>Avatar</entity>?",
        return_dict=True,
    )
)

# __start__ i'm good. what kind of movies do you like? __end__
#  - Avatar
#  - Trojan War
#  - The Patriot

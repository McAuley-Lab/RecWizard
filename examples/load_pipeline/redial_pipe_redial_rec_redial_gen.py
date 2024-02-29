from recwizard.modules.redial.modeling_redial_rec import RedialRec, RedialRecTokenizer
from recwizard.modules.redial.modeling_redial_gen import RedialGen, RedialGenTokenizer

from recwizard.pipelines.redial_only import RedialOnlyPipeline, RedialOnlyConfig

gen_repo = "../local_repo/redial-gen"
rec_repo = "../local_repo/redial-rec"

# test after pushed to hub

# load rec
rec_tokenizer = RedialRecTokenizer.from_pretrained(rec_repo)
rec_module = RedialRec.from_pretrained(rec_repo)

## load gen
gen_tokenizer = RedialGenTokenizer.from_pretrained(gen_repo)
gen_module = RedialGen.from_pretrained(gen_repo)

# load redial pipeline
pipeline = RedialOnlyPipeline(
    config=RedialOnlyConfig(),
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

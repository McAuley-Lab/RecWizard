from recwizard.modules.kgsf.modeling_kgsf_rec import KGSFRec, KGSFRecTokenizer
from recwizard.modules.kgsf.modeling_kgsf_gen import KGSFGen, KGSFGenTokenizer

from recwizard.pipelines.trivial import TrivialConfig, TrivialPipeline

gen_repo = "../local_repo/kgsf-gen"
rec_repo = "../local_repo/kgsf-rec"

# test after pushed to hub

# load rec
rec_tokenizer = KGSFRecTokenizer.from_pretrained(rec_repo)
rec_module = KGSFRec.from_pretrained(rec_repo)

## load gen
gen_tokenizer = KGSFGenTokenizer.from_pretrained(gen_repo)
gen_module = KGSFGen.from_pretrained(gen_repo)

# load tirival pipeline
pipeline = TrivialPipeline(
    config=TrivialConfig(),
    rec_module=rec_module,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    gen_tokenizer=gen_tokenizer,
)

print(pipeline.response("I like <entity>Avatar</entity>, and you?", return_dict=False))

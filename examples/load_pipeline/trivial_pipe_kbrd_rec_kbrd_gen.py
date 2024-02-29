from recwizard.modules.kbrd import KBRDGen, KBRDRec, KBRDGenTokenizer, KBRDRecTokenizer
from recwizard.pipelines.trivial import TrivialConfig, TrivialPipeline

gen_repo = "../local_repo/kbrd-gen"
rec_repo = "../local_repo/kbrd-rec"

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

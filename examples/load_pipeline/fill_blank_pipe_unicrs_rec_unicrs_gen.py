from recwizard.modules.unicrs.modeling_unicrs_rec import UnicrsRec, UnicrsRecTokenizer
from recwizard.modules.unicrs.modeling_unicrs_gen import UnicrsGen, UnicrsGenTokenizer

from recwizard.pipelines.fill_blank import FillBlankConfig, FillBlankPipeline

gen_repo = "../local_repo/unicrs-gen"
rec_repo = "../local_repo/unicrs-rec"

# test after pushed to hub

# load rec
rec_tokenizer = UnicrsRecTokenizer.from_pretrained(rec_repo)
rec_module = UnicrsRec.from_pretrained(rec_repo)

## load gen
gen_tokenizer = UnicrsGenTokenizer.from_pretrained(gen_repo)
gen_module = UnicrsGen.from_pretrained(gen_repo)

# load tirival pipeline
pipeline = FillBlankPipeline(
    config=FillBlankConfig(),
    rec_module=rec_module,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    gen_tokenizer=gen_tokenizer,
)

print(pipeline.response("I like <entity>Avatar</entity>, and you?", return_dict=True))

# __start__ i'm good. what kind of movies do you like? __end__
#  - Avatar
#  - Trojan War
#  - The Patriot

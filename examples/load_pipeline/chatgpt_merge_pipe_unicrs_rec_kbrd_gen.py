from recwizard.modules.unicrs.modeling_unicrs_rec import UnicrsRec, UnicrsRecTokenizer
from recwizard.modules.kbrd.modeling_kbrd_gen import KBRDGen, KBRDGenTokenizer

from recwizard.pipelines.chatgpt_merge import ChatgptMergeConfig, ChatgptMergePipeline

rec_repo = "../local_repo/unicrs-rec"
gen_repo = "../local_repo/kbrd-gen"

# test after pushed to hub

# load rec
rec_tokenizer = UnicrsRecTokenizer.from_pretrained(rec_repo)
rec_module = UnicrsRec.from_pretrained(rec_repo)

## load gen
gen_tokenizer = KBRDGenTokenizer.from_pretrained(gen_repo)
gen_module = KBRDGen.from_pretrained(gen_repo)

# load tirival pipeline
pipeline = ChatgptMergePipeline(
    config=ChatgptMergeConfig(),
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

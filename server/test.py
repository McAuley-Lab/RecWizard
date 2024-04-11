from recwizard import (
    FillBlankConfig,
    FillBlankPipeline,
    UnicrsGen,
    UnicrsRec,
)

unicrs = FillBlankPipeline(
    config=FillBlankConfig(),
    gen_module=UnicrsGen.from_pretrained("../../local_repo/unicrs-gen"),
    rec_module=UnicrsRec.from_pretrained("../../local_repo/unicrs-rec"),
)

print(unicrs.response("I like <entity>Avatar</entity>, and you?", return_dict=True))

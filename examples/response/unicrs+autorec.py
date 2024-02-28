from recwizard.utils import DeviceManager
from recwizard.modules.unicrs import UnicrsGen, UnicrsGenTokenizer
from recwizard.modules.redial import RedialRec, RedialRecTokenizer
from recwizard.pipelines.fill_blank.configuration_fill_blank import FillBlankConfig
from recwizard.pipelines.fill_blank.modeling_fill_blank import FillBlankPipeline

if __name__ == "__main__":
    model = FillBlankPipeline(
        config=FillBlankConfig(),
        gen_module=UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial"),
        gen_tokenizer=UnicrsGenTokenizer.from_pretrained("recwizard/unicrs-gen-redial"),
        rec_module=RedialRec.from_pretrained("recwizard/redial-rec"),
        rec_tokenizer=RedialRecTokenizer.from_pretrained("recwizard/redial-rec"),
    ).to(DeviceManager.device)

    query = (
        "System: Hello!"
        "<sep>User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>."
        " Would you please recommend me some other movies?"
    )

    resp = model.response(query, return_dict=True)

    # query = ('User: Hello!'
    #          '<sep>System: Hello, what kind of movies do you like?'
    #          '<sep>User: I like movie!'
    #          )

    print(query)
    print(resp["output"])
    print(resp)

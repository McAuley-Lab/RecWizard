from recwizard.modules.unicrs import UnicrsGen, UnicrsRec, UnicrsRecTokenizer, UnicrsGenTokenizer
from recwizard.pipelines.fill_blank.configuration_fill_blank import FillBlankConfig
from recwizard.pipelines.fill_blank.modeling_fill_blank import FillBlankPipeline

if __name__ == '__main__':
    model = FillBlankPipeline(
        config=FillBlankConfig(),
        gen_module=UnicrsGen.from_pretrained('recwizard/unicrs-gen-redial'),
        # gen_tokenizer=UnicrsGenTokenizer.from_pretrained('recwizard/UnicrsGen-redial'),
        rec_module=UnicrsRec.from_pretrained('recwizard/unicrs-rec-redial'),
        # rec_tokenizer=UnicrsRecTokenizer.from_pretrained('recwizard/UnicrsRec-redial')
    ).to('cuda:0')
    query = ('System: Hello!'
             '<sep>User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>.'
             ' Would you please recommend me some other movies?'
             )

    # print(query)
    result = model.response(query, return_dict=True)
    print(query)
    print(result['output'])

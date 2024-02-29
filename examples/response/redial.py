from recwizard.modules.redial import RedialGen, RedialRec
from recwizard.tokenizers import MultiTokenizer
from recwizard.utils import load_json_file_from_dataset, init_deterministic, EntityLink, DeviceManager
from recwizard.pipelines import RedialPipelineConfig, RedialPipeline

init_deterministic(42)
# DeviceManager.initialize('cpu')
query = (
    "User: Hi I am looking for a movie like <entity>Super Troopers (2001)</entity>"
    "<sep>System: You should watch <entity>Police Academy (1984)</entity>"
    "<sep>User: Is that a great one? I have never seen it. I have seen <entity>American Pie</entity> I mean <entity>American Pie (1999)</entity>"
    "<sep>System: Yes <entity>Police Academy (1984)</entity> is very funny and so is <entity>Police Academy 2: Their First Assignment (1985)</entity>"
    "<sep>User: What is some other options?"
    # '<sep>System: yes you will enjoy them'
    # '<sep>Usesr: I appreciate your time. I will need to check those out. Are there any others you would recommend?'
)

query1 = (
    "System: Hello!"
    "<sep>User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>."
    " Would you please recommend me some other movies?"
)


gen_tokenizer = MultiTokenizer.from_pretrained("Kevin99z/redial-gen")
rec_tokenizer = MultiTokenizer.from_pretrained("Kevin99z/redial-rec")

gen_module = RedialGen.from_pretrained("recwizard/redial-gen")
rec_module = RedialRec.from_pretrained("recwizard/redial-rec")

model = RedialPipeline(
    config=RedialPipelineConfig(),
    gen_tokenizer=gen_tokenizer,
    rec_tokenizer=rec_tokenizer,
    gen_module=gen_module,
    rec_module=rec_module,
)

model = model.to("cuda:0")

result = model.response(query, return_dict=True)

print(query)
print(result)

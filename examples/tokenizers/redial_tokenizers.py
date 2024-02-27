from recwizard.modules.redial import RedialGen, RedialRec
from recwizard.utility import load_json_file_from_dataset
dataset = 'redial'
vocab = load_json_file_from_dataset(dataset, 'vocab.json')
id2entity = load_json_file_from_dataset(dataset, 'id2entity.json')
        # prepare tokenizer
from transformers import AutoTokenizer
from recwizard.tokenizers import NLTKTokenizer, EntityTokenizer, MultiTokenizer
from tokenizers.normalizers import Replace, Sequence
sen_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")
sen_tokenizer.backend_tokenizer.normalizer  = Sequence(
    [Replace("<entity>", ""), Replace("</entity>", ""), Replace("_", " ")] 
)
rnn_tokenizer = NLTKTokenizer(
    unk_token='<unk>', pad_token='<pad>', bos_token='<s>', eos_token='</s>',
    vocab=  {word: i for i, word in enumerate(vocab)}
)

query = query = (
    "User: Hi I am looking for a movie like <entity>Super Troopers (2001)</entity>"
    "<sep>System: You should watch <entity>Police Academy (1984)</entity>"
    "<sep>User: Is that a great one? I have never seen it. I have seen <entity>American Pie</entity> I mean <entity>American Pie (1999)</entity>"
    "<sep>System: Yes <entity>Police Academy (1984)</entity> is very funny and so is <entity>Police Academy 2: Their First Assignment (1985)</entity>"
    "<sep>User: What is some other options?"
    # '<sep>System: yes you will enjoy them'
    # '<sep>Usesr: I appreciate your time. I will need to check those out. Are there any others you would recommend?'
)

gen_tokenizer = MultiTokenizer(tokenizers={"dialogue": rnn_tokenizer, 
                                "entity": EntityTokenizer(
                                    vocab={v: int(k) for k, v in id2entity.items()}
                                ),
                                "sen_encoding": sen_tokenizer
                                }, tokenizer_key_for_decoding="dialogue")


gen_module = RedialGen.from_pretrained("recwizard/redial-gen")
# gen_tokenizer.save_pretrained('Kevin99z/redial-gen')
# gen_tokenizer = MultiTokenizer.from_pretrained('Kevin99z/redial-gen')
result = gen_module.response(query, gen_tokenizer)
print(result)

rec_tokenizer = MultiTokenizer(
    tokenizers={
                "entity": EntityTokenizer(
                    vocab={v: int(k) for k, v in id2entity.items()}
                ),
                "sen_encoding": sen_tokenizer
                }, tokenizer_key_for_decoding="entity"
)

rec_module = RedialRec.from_pretrained("recwizard/redial-rec")
# rec_tokenizer.save_pretrained('Kevin99z/redial-rec')
# rec_tokenizer = MultiTokenizer.from_pretrained('Kevin99z/redial-rec')
result = rec_module.response(query, rec_tokenizer)
print(result)

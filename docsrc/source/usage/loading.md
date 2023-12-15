## Load Existing Models

### 1. CRS Models as a Whole

After installing [RecWizard](./index), we quickly load an existing Converstional Recommender Bot using a model name as follows:

```python
from recwizard import AutoModel
unicrs_model = AutoModel.from_pretrained('Unicrs-redial')

unicrs_model.response("Recommend me some popular and classic movies, \
                    I like <entity> Titanic </entity>.")
```

We input the conversation history and get the results from the pre-trained [UniCRS](https://arxiv.org/abs/2206.09363) model on [ReDIAL](https://proceedings.neurips.cc/paper_files/paper/2018/file/800de15c79c8d840f4e78d3af937d4d4-Paper.pdf) dataset.

```bash
'I would like to recommend The Shawshank Redemption (1994) to you!'
```

We can find more shared CRS model cards in our [resource section](resource/model_zoo).


### 2. CRS Models at Module Level

Apart from loading models with an one-liner similar to the module-level example above, it is also flexible to create model variants by using different module-level combinations. Here we specify a [UniCRS](https://arxiv.org/abs/2206.09363) variant by using a new [ChatGPT](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)-based generator module to build a `unicrs_model_variant`:

```python
from recwizard import FillBlankPipeline, FillBlankConfig, UnicrsRec, ChatgptGen

unicrs_model_variant = recwizard.AutoModel(
    config=FillBlankConfig(), 
    rec_module=UnicrsRec.from_pretrained('RecWizard/unicrs-rec-redial'),
    gen_module=ChatgptGen.from_pretrained('RecWizard/chatgpt-gen-fillblank')
)
```

Then, this initialized `unicrs_model_variant` can be used in a similar way as Case 1:

```python
unicrs_model_variant.response("Recommend me some popular and classic movies, \
                I like <entity> Titanic </entity>.")
```

We have the response from the UniCRS variant with ChatGPT generator module soon:

```bash
"Sure, if you enjoyed 'Titanic,' you might like these other popular and 
     classic movie like The Shawshank Redemption (1994)"
```

So, this example shows that we can load the CRS modules first (they are typically recommender modules and generator modules). Then, feed the instantiated modules into a higher-level RecWizard model, where the model knows how to manage the usages and the results from different modules. If you are willing to know more about the model / module concepts in RecWizard, please check this [concept section](./).

### 3. Single CRS Modules

Specially, as shown in Case 2, we can load CRS modules first. Also, we can use the CRS (recommender or generator) module independently, let us use the recommender module from ReDIAL as an example:

```python
from recwizard import RedialRec, RedialRecTokenizer

# load redial recommender module
redial_rec_module = RedialRec.from_pretrained('RecWizard/redial-rec')

# load redial recommender module tokenizer (convert text data to tensor data)
redial_rec_tokenizer = RedialRecTokenizer.from_pretrained('RecWizard/redial-rec')
```

#### 3.1. Deal with Text Data 

We can play with the module directly using natural language inputs, where if you wrap all the mentioned item titles with `<entity>` labels, such as `<entity>Titanic</entity>`, the `Titanic` mention will be mapped to the corresponding *item id*. Otherwise, the `Titanic` mention will be mapped to normal corresponding *word id*:

```python
redial_rec_module.response(
    raw_input="Recommend me some popular and classic movies, I like <entity>Titanic</entity>.",
    tokenizer=redial_rec_tokenizer,
    topk=5,
)
```

Then we obtain the top-5 results from the ReDIAL recommender module:

```bash
[
 'The Big Sick (2017)',
 'Bad Moms (2016)',
 'Dumb and Dumber (1994)',
 '21 Jump Street (2012)',
 'Star Wars: Episode VIII – The Last Jedi (2017)'
]
```

#### 3.2. Deal with Tensor Data 

All the above examples are using the `.response` method to deal with **text data**, but for the internal mechanism within (recommender or generator) modules, they are typically dealing with **tensor data**. So, a lower-level control is also provided if the users are curious about the **tensor data**. Bascially, `response` is equivalent to "tokenzier `encode` -> module `forward` -> tokenizer `decode`.

##### 3.2.1. Tokenzier `encode`

```python 
raw_input='User: Hi I am looking for a movie like <entity>Super Troopers (2001)</entity>'
tensor_input = redial_rec_tokenizer([raw_input])
tensor_input
```

```bash
{'attention_mask': tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]), 'senders': tensor([[1]]), 'input_ids': tensor([[[    0, 30086,    38,   524,   546,    13,    10,  1569,   101,  1582,
           6354, 18158,     2]]]), 'movieIds': tensor([[1901]]), 'conversation_lengths': tensor([1]), 'movie_occurrences': [tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.]]])]}
```

##### 3.2.2. Module `forward` 

```python
tensor_output = redial_rec_module(**tensor_input) # equivalent to redial_rec_module.forward(**tensor_input)
print(tensor_output)
```

```bash
tensor([[[-20.4876, -13.0947, -20.1804,  ..., -20.6572, -21.6375, -20.9953]]],
       grad_fn=<AsStridedBackward0>)
```

##### 3.2.2. Tokenizer `decode`

```python
movieIds = tensor_output.topk(k=3, dim=-1).indices[:, -1,
                   :].tolist()
tokenizer.decode(movieIds[0])
```

```bash
['The Big Sick (2017)', 'Bad Moms (2016)', 'Dumb and Dumber (1994)']
```

Additionally, if you are interestd in more existing modules, please check our [resource section](resource/module_zoo).

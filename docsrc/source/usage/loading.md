## Load Existing Models

### 1. CRS Models as a Whole

After installing [🤖️ RecBOT](./index), we quickly load an existing Converstional Recommender Bot using a model name as follows:

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
from recwizard import AutoModel, AutoModule, AutoToeknizer

unicrs_model_variant = recwizard.AutoModel(
    config=recwizard.FillBlankConfig(), 
    rec_module=AutoModule.from_pretrained('UnicrsRec-redial')
    rec_tokenizer=AutoTokenizer.from_pretrained('UnicrsRec-redial')
    gen_module=AutoModule.from_pretrained('UnicrsGen-chatgpt')
    gen_tokenizer=AutoTokenizer.from_pretrained('UnicrsGen-chatgpt'),
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

So, this example shows that we can load the CRS modules first (they are typically recommender modules and generator modules). Then, feed the instantiated modules into a higher-level RecBOT model, where the model knows how to manage the usages and the results from different modules. If you are willing to know more about the model / module concepts in RecBOT, please check this [concept section](./).

### 3. Single CRS Modules

Specially, as shown in Case 2, we can load CRS modules first. Also, we can use the CRS (recommender or generator) module independently, let us use the recommender module from UniCRS as an example:

```python
from recwizard import AutoModule, AutoToeknizer

# load unicrs recommender module
unicrs_rec=AutoModule.from_pretrained('UnicrsRec-redial')

# load unicrs recommender module tokenizer (convert text data to tensor data)
unicrs_rec_tokenizer=AutoTokenizer.from_pretrained('UnicrsRec-redial')
```

#### 3.1. Deal with Text Data 

We can play with the module directly using natural language inputs, where if you wrap all the mentioned item titles with `<entity>` labels, such as `<entity> Titanic </entity>`, the `Titanic` mention will be mapped to the corresponding *item id*. Otherwise, the `Titanic` mention will be mapped to normal corresponding *word id*:

```python
unicrs_rec.response(
    raw_input="Recommend me some popular and classic movies, I like <entity> Titanic </entity>.",
    tokenizer=unicrs_rec_tokenizer,
    topk=5,
)
```

Then we obtain the top-5 results from the UniCRS recommender module:

```bash
"<entity> The Shawshank Redemption (1994) </entity>
 <entity> The Godfather (1972) </entity>
 <entity> The Dark Knight (2008) </entity>
 <entity> The Godfather: Part II (1974) </entity> 
 <entity> The Lord of the Rings: The Return of the King (2003) </entity>"
```

#### 3.2. Deal with Tensor Data 

All the above examples are using the `.response` method to deal with **text data**, but for the internal mechanism within (recommender or generator) modules, they are typically dealing with **tensor data**. So, a lower-level control is also provided if the users are curious about the **tensor data**. Bascially, `response` is equivalent to "tokenzier `encode` -> module `forward` -> tokenizer `decode`.

##### 3.2.1. Tokenzier `encode`

```python 
raw_input="Recommend me some popular and classic movies, I like <entity> Titanic </entity>."
tensor_input = unicrs_rec_tokenizer.encode(raw_input)
```

```bash
TODO: add the example tensor data as the outputs from "unicrs_rec_tokenizer.encode(raw_input)"
```

##### 3.2.2. Module `forward` 

```python
tensor_ouput = unicrs_rec(**tensor_input) # equivalent to unicrs_rec.forward(**tensor_input)
```

```bash
TODO: add the example tensor data as the outputs from "unicrs_rec.forward(**tensor_input)"
```

##### 3.2.2. Tokenizer `decode`

```python
text_ouput = unicrs_rec_tokenizer.decode(tensor_output)
```

```bash
TODO: add the example tensor data as the outputs from "unicrs_rec_tokenizer.decode(tensor_output)"
```

Additionally, if you are interestd in more existing modules, please check our [resource section](resource/module_zoo).
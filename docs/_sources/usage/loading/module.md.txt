#### 2. CRS Models at Module Level

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
#### 1. CRS Models as a Whole

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

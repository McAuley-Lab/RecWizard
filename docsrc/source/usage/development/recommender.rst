2. NEW Recommender
^^^^^^^^^^^^^^^^^^^

A NEW recommender can be used individually in this way:

2.1. Raw text input and output
******************************

.. code-block:: 

    System: Hello!<sep>
    User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>. 
    Would you please recommend me some other movies?


.. code-block:: python

    ['21 Bridges (2019)', 'The Conjuring (2013)', 'The Exorcist (1973)']

2.2. Tensor input and output
****************************

.. code-block:: python

    # inputs
    {'input_ids': tensor([[1, 2]]), 'attention_mask': tensor([[True, True]])}

.. code-block:: python

    # logits
    tensor([[ 4.3385,  7.7007,  6.6780, -1.6603, -1.5623,  3.8379,  2.0713,  0.2687]], grad_fn=<SumBackward1>)

2.3. Implementation of NEW recommender
**************************************

**2.3.1. Recommender Configuration: NEWRecConfig**

.. code-block:: python

    from recwizard.configuration_utils import BaseConfig

    class NEWRecConfig(BaseConfig):
        """Configuration class to sotre the
        configuration of the NEW recommender."""

        def __init__(self, n_items: int = None, dim: int = None, **kwargs):
            super().__init__(**kwargs)

            self.n_items = n_items
            self.dim = dim

    # use it!
    config = NEWRecConfig(n_items=8, dim=10)

**2.3.2. Recommender Tokenizer: NEWRecTokenizer**

.. code-block:: python

    from typing import List

    from recwizard.tokenizer_utils import BaseTokenizer
    from recwizard.utility.utils import WrapSingleInput


    class NEWRecTokenizer(BaseTokenizer):
        """Tokenizer class for the NEW recommender."""

        @WrapSingleInput
        def decode(self, ids, *args, **kwargs) -> List[str]:
            """Decode a list of token ids into a list of strings.
            Args:
                ids (List[int]): list of token ids to decode;
            Returns:
                List[str]: list of decoded strings;
            """
            return [self.id2entity[id] for id in ids if id in self.id2entity]

        def __call__(self, *args, **kwargs):
            """Tokenize a string into a list of token ids."""
            kwargs.update(return_tensors="pt", padding=True, truncation=True)
            return super().__call__(*args, **kwargs)

    # use it!
    tokenizer = NEWRecTokenizer(id2entity={
        0: '21 Bridges (2019)',
        1: 'The Shining (1980)',
        2: 'Annabelle (2014)',
        3: 'The Conjuring (2013)',
        4: 'The Exorcist (1973)',
        5: 'The Conjuring 2 (2016)',
        6: 'The Nun (2018)',
        7: 'X men (2019)',
    })

**3.3.3. Recommender Module: NewRec**

.. code-block:: python

    import torch

    from recwizard.module_utils import BaseModule
    from transformers.utils import ModelOutput


    class NEWRec(BaseModule):
        """NEW is a module that implements the NEW recommender."""

        config_class = NEWRecConfig
        tokenizer_class = NEWRecTokenizer

        def __init__(self, config: NEWRecConfig, **kwargs):
            super().__init__(config, **kwargs)

            self.embeds = torch.nn.Embedding(config.n_items, config.dim)

        def forward(self, input_ids, attention_mask=None):
            """Forward pass of the NEW recommender."""
            
            embeds = self.embeds(input_ids)
            avg_embeds = embeds.sum(dim=1) / (attention_mask.sum(dim=-1, keepdim=True) + 1e-8)
            logits = (self.embeds.weight * avg_embeds.unsqueeze(1)).sum(dim=-1)
            return ModelOutput({"rec_logits": logits})
        
        @WrapSingleInput
        def response(self, raw_input, tokenizer, return_dict=False, topk=3):
            """Generate response from the NEW recommender."""

            # convert text input to tensor input
            entities = tokenizer(raw_input)['entities'].to(self.device)
            inputs = {
                "input_ids": entities,
                "attention_mask": entities != tokenizer.pad_entity_id,
            }
            
            # recommend top-k items
            logits = self.forward(**inputs)["rec_logits"]
            print(inputs, logits)
            logits[torch.arange(logits.size(0)), entities] = float("-inf")
            recommended = logits.topk(topk).indices.tolist()
            output = tokenizer.batch_decode(recommended)

            # return the output
            if return_dict:
                return {
                    "output": output, 
                    "input": raw_input, 
                    "recommended": recommended
                }
            return output

    # use it!

    model = NEWRec(config)

    query = ('System: Hello!'
            '<sep>User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>.'
            'Would you please recommend me some other movies?'
            )

    resp = model.response(
        raw_input=query, 
        tokenizer=tokenizer,
        return_dict=True
    )
    
The complete implementation is in `examples/develop_model/new_recommender.py`.

## Overview

This is an overview about How to create new `RecWizard` modules (such as generators, recommenders and processors) or models (such as UniCRS models).
In this tutorial, we aim to create and share a new model, called, NEW trained on [`INSPIRED`]() to go through the journey of contributing to `RecWizard`.

### 1. Tutorial Setup

We create and share the NEW as a new conversational recommender in three steps:

1. Create and train a NEW recommender;
2. Create and train a NEW generator;
3. Assemble the NEW recommender and generator under the NEW model;
4. Add interface features to NEW model;
5. More details about how to contribute to RecWizard codebase.

Let us look at the desired NEW model in advance, and check the detailed implementation steps in the corresponding sections:

#### 1.1 NEW Recommender [[details]](./module)

A NEW recommender can be used individually in this way:

1. Raw text input and output
    ```text
    System: Hello!<sep>
    User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>. 
    Would you please recommend me some other movies?
    ```

    ```output
    ['21 Bridges (2019)', 'The Conjuring (2013)', 'The Exorcist (1973)']
    ```

2. Tensor input and output
    ```python 
    # inputs
    {'input_ids': tensor([[1, 2]]), 'attention_mask': tensor([[True, True]])}
    ```

    ```python
    # logits
    tensor([[ 4.3385,  7.7007,  6.6780, -1.6603, -1.5623,  3.8379,  2.0713,  0.2687]], grad_fn=<SumBackward1>)
    ```

3. We need to implement three components for this NEW recommender:

    3.1. Recommender Configuration: `NEWRecConfig`
    
    ```python 
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
    ```

    3.2. Recommender Tokenizer: `NEWRecTokenizer`

    ```python
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
    ```

    3.3. Recommender Module: `NewRec`

    ```python
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
    ```
    
The complete implementation is in `examples/develop_model/new_recommender.py`.

#### 1.2 NEW Generator [[details]](./module)

A NEW recommender can be used individually in this way:

1. Raw text input and output
    ```python

    ```

2. Tensor input and output
    ```python 

    ```

3. We need to implement three components for this NEW generator:

    3.1. Generator Configuration: `NEWGenConfig`

    ```python
    from recwizard.configuration_utils import BaseConfig

    class NEWGenConfig(BaseConfig):
        """Configuration class to sotre the
        configuration of the NEW Generator."""

        def __init__(
            self, base_model: str = "microsoft/DialoGPT-small", n_items: int = None, max_gen_len=100, **kwargs
        ):
            super().__init__(**kwargs)

            self.base_model = base_model
            self.n_items = n_items
            self.max_gen_len = max_gen_len

    # use it! 
    config = NEWGenConfig(base_model='microsoft/DialoGPT-small', n_items=8)
    ```

    3.2. Generator Tokenizer: `NEWGenTokenizer`
    
    ```python
    from recwizard.tokenizer_utils import BaseTokenizer


    class NEWGenTokenizer(BaseTokenizer):
        """Tokenizer class for the NEW generator."""

        def __call__(self, *args, **kwargs):
            """Tokenize a string into a list of token ids."""
            kwargs.update(return_tensors="pt", padding=True, truncation=True)
            return super().__call__(*args, **kwargs)

    # use it!
    word_tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
    word_tokenizer.pad_token = word_tokenizer.eos_token
    tokenizer = NEWGenTokenizer(
        tokenizers=[word_tokenizer],
        id2entity={
            0: "21 Bridges (2019)",
            1: "The Shining (1980)",
            2: "Annabelle (2014)",
            3: "The Conjuring (2013)",
            4: "The Exorcist (1973)",
            5: "The Conjuring 2 (2016)",
            6: "The Nun (2018)",
            7: "X men (2019)",
        },
    )
    ```

    3.3. Generator Module: `NEWGen`

    ```python

    import torch

    from recwizard.module_utils import BaseModule
    from recwizard.utility.utils import WrapSingleInput
    from transformers import GPT2LMHeadModel


    class NEWGen(BaseModule):
        """NEW is a module that implements the NEW generator."""

        config_class = NEWGenConfig
        tokenizer_class = NEWGenTokenizer

        def __init__(self, config: NEWGenConfig, **kwargs):
            super().__init__(config, **kwargs)

            self.gpt2_model = GPT2LMHeadModel.from_pretrained(config.base_model)
            self.entity_embeds = torch.nn.Embedding(
                config.n_items, self.gpt2_model.config.n_embd
            )
            self.max_gen_len = config.max_gen_len

        def generate(self, context, entities, attention_mask, **kwargs):
            """Forward pass of the NEW generator."""

            embeds = self.entity_embeds(entities)
            avg_embeds = embeds.sum(dim=1, keepdim=True) / (
                attention_mask.sum(dim=-1, keepdim=True) + 1e-8
            )
            text_embeds = self.gpt2_model.transformer.wte(context["input_ids"])
            inputs_embeds = torch.cat([avg_embeds, text_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(*avg_embeds.shape[:2]).to(avg_embeds.device),
                    context["attention_mask"],
                ],
                dim=1,
            )

            return self.gpt2_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_len,
                return_dict_in_generate=True,
                **kwargs,
            )

        @WrapSingleInput
        def response(self, raw_input, tokenizer, return_dict=False):
            """Generate response from the NEW generator."""

            inputs = tokenizer(raw_input)
            context = {
                "input_ids": inputs["input_ids"].to(self.device),
                "attention_mask": inputs["attention_mask"].to(self.device),
            }

            # convert text input to entity input
            entities = inputs["entities"].to(self.device)
            attention_mask = entities != tokenizer.pad_entity_id

            # generate response
            generated = self.generate(
                context=context, entities=entities, attention_mask=attention_mask
            )
            output = tokenizer.batch_decode(generated.sequences)

            # return the output
            if return_dict:
                return {"output": output, "input": raw_input, "generated": generated}
            return output

    # use it!
    model = NEWGen(config)

    query = (
        "System: Hello!"
        "<sep>User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>."
        "Would you please recommend me some other movies?"
    )

    resp = model.response(raw_input=query, tokenizer=tokenizer, return_dict=True)
    ```

#### 1.3 NEW Pipeline [[details]](./model)

NEW Model can define the logitics of how to use NEW recommender and generator.

1. Raw text input and output
    ```python

    ```

2. Tensor input and output
    ```python 

    ```

3. Create this high-level NEW Pipeline

    3.1. Create NEW Pipeline Configuration: `NEWConfig`

    ```python 
    from recwizard.configuration_utils import BaseConfig

    class NEWConfig(BaseConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    ```

    3.2. Create NEW Pipeline: `NEWPipeline`

    ```python
    from recwizard.model_utils import BasePipeline

    class NEWPipeline(BasePipeline):
        config_class = NEWConfig

        def forward(self, input_ids, attention_mask, labels=None, **kwargs):
            raise NotImplementedError

        @monitor
        def response(
            self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs
        ):
            rec_args = rec_args or {}
            gen_args = gen_args or {}
            rec_output = self.rec_module.response(
                query, tokenizer=self.rec_tokenizer, return_dict=True, **rec_args
            )

            query_condition_on_rec = [
                q + "System: I recommend " + self.rec_tokenizer.decode(r) + "because"
                for q, r in zip(query, rec_output["recommended"])
            ]

            gen_output = self.gen_module.response(
                query_condition_on_rec,
                tokenizer=self.gen_tokenizer,
                return_dict=True,
                **gen_args,
            )
            if return_dict:
                return {
                    "rec_logits": rec_output["logits"],
                    "gen_logits": gen_output["logits"],
                    "rec_output": rec_output["output"],
                    "gen_output": gen_output["output"],
                }

            return gen_output["output"][0] + "\n - " + "\n - ".join(rec_output["output"])
    ```

#### 1.4 NEW Interactive Interface [[details]](./model)

We can use `recwizard.Monitor` to define and launch the interactive interface after building the model.

```python 

```
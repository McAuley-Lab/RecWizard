3.  NEW Generator
^^^^^^^^^^^^^^^^^

A NEW recommender can be used individually in this way:

3.1. Raw text input and output
******************************

.. code-block:: 

    System: Hello!<sep>
    User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>.
    Would you please recommend me some other movies?

3.2. Tensor input and output
****************************

.. code-block:: python

    # TODO


3.3. Implementation of NEW generator
************************************

**3.3.1. Generator Configuration: NEWGenConfig**

.. code-block:: python

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

**3.3.2. Generator Tokenizer: NEWGenTokenizer**

.. code-block:: python

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

**3.3.3. Generator Module: NEWGen**

.. code-block:: python

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

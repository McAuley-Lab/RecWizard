4. NEW Pipeline
^^^^^^^^^^^^^^^

NEW Model can define the logitics of how to use NEW recommender and generator.

4.1. Raw text input and output
******************************

.. code-block:: 

    System: Hello!<sep>
    User: Hi. I like horror movies, such as <entity>The Shining (1980)</entity> and <entity>Annabelle (2014)</entity>.
    Would you please recommend me some other movies?


4.2. Tensor input and output
****************************

.. code-block:: python

    # TODO


4.3.  Implementation of NEW Pipeline
************************************

**4.3.1. Create NEW Pipeline Configuration: NEWConfig**

.. code-block:: python

    from recwizard.configuration_utils import BaseConfig

    class NEWConfig(BaseConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


**4.3.2. Create NEW Pipeline: NEWPipeline**

.. code-block:: python

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

5. NEW Interactive Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use `recwizard.Monitor` to define and launch the interactive interface after building the model.

Step 1: Add @monitor decorator to your model/pipeline's response function: 
.. code-block:: python

    from recwizard.model_utils import BasePipeline
  + from recwizard import monitor

    class NEWPipeline(BasePipeline):
        ...

    +   @monitor
        def response(
            self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs
        ):

        ...


Step 2: Use `monitoring` context with your model/pipeline:

.. code-block:: python
    
    newPipeline = NEWPipeline(
                config=NEWConfig(),
                gen_module=yourGenModule,
                rec_module=yourRecModule,
            )
    with monitoring(mode="debug") as m:
        response = newPipeline.response(
            query, return_dict=True, rec_args=rec_args, gen_args=gen_args
        )
        logger.info(f"Response: {response}")
        logger.info(f"Details: {response.graph}")

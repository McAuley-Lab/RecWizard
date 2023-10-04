First Recwizard pipeline
------------------------
Let's start by building a ``ExpansionPipeline`` using a ChatGPT-based generator with an AutoRec-based recommender [1]_.

1. Setup the pipeline
---------------------

.. code-block:: python

    from recwizard import ExpansionConfig, ExpansionPipeline
    from recwizard import ChatgptGen, RedialRec

    pipeline = ExpansionPipeline(
      ExpansionConfig(),
      rec_module=RedialRec.from_pretrained('recwizard/redial-rec'),
      gen_module=ChatgptGen.from_pretrained('recwizard/chatgpt-expansion')
    )

2. Format the input
-------------------

You would want to format your dialogue history like this before passing it to the pipeline:

.. code-block:: python

    context = "<sep>".join([
      "User: Hello!",
      "System: Hello, I have some movie ideas for you. Have you watched the movie <entity>Forever My Girl (2018)</entity> ?",
      "User: Looking for movies in the comedy category. I like Adam Sandler movies like <entity>Billy Madison (1995)</entity> Oh no is that good?"
    ])

Currently, the formatting contains 3 aspects:

1. We prepend the sender of the message to the beginning of the sentence. Either "User: " or "System: "
2. We join the utterances with the <sep> token, which can be handled differently by different tokenizers
3. (Optional) We mark the entities in the text manually so that the named entities can be extracted by the downstream modules.

3. Get Response!
----------------

.. code-block:: python

    print(pipeline.response(context))

.. [1] Raymond Li, Samira Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, Chris Pal. "Towards Deep Conversational Recommendations". 2019. arXiv:1812.07617 [cs.LG].
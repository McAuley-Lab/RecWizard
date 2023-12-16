.. _getting-started:

Concepts
--------

1. Design Principles
^^^^^^^^^^^^^^^^^^^^

We abstract a CRS model to two levels as shown in the main figure:

- **Module Level**: we typically provide a `rectbot.BaseModule` for making recommendations and another `rectbot.BaseModule` for generating natural-language responses, we can further introduce processor `rectbot.BaseModule` to extract important information (e.g., entity linking) from users' raw features;
- **Pipeline Level**: we treat a `rectbot.BasePipline` is a high-level manager to decide when and how to call the lowe-level modules, and how to combine the results from such modules.

By default, all the modules are communicating in natural-language (e.g., **text data**) formats, which is the key to make all the modules as replaceable as possible; we still expose low-level methods for developers to define module communications via **tensor data** in a flexible and differentiable way.

2. Features
^^^^^^^^^^^

With the seamless compatibility of the Huggingface, stemming from our design principles to "let modules communicate in natural language" while "exposing low-level APIs to users", RecWizard shows the following properties:

- **Modular**: CRS model = (lower) *module level* + (higher) *pipline level*.
- **Portable**: Share or load CRS modules and pipelines from Huggingface Hub! 
- **Interactive**: Chat with CRS models via our interactive interfaces. 
- **LLMs-Friendly**: Implemented LLMs as different roles in CRS models.

3. FAQs
^^^^^^^
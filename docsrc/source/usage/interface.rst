Interact with Models
----------------------

In `RecWizard <./index>`_, users can interact with the existing CRS models using our interactive interface.

1. Launch Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ TODO from Tanmay

2. INFO Mode
^^^^^^^^^^^^^
In this mode, users can normally chat with the system asking for natural language responses or recommended items from the RecWizard model. 

2.1. When To Use 
"""""""""""""""""

- **Case 1**: Demonstrating the final RecWizard model; 

- **Case 2**: Inviting users for human evaluation.

2.2. How To Use 
"""""""""""""""""

2.2.1. Chatting
*****************

The chatting feature in this mode allows users to engage in interactive conversations with our recwizard model by simply typing in their messages. 

If the users need to explicitly mention some items, for example, a movie. The way to do so is type in the item name with the <entity> paired labels. For example, `<entity> The Matrix </entity>`.

2.2.2. Histories Exportation
*****************************

Our users can also export the conversational recommendation histories, for example, after exportaion, sharing the chat logs, the recommended items, and the user feedbacks with others for further analysis.

3. DEBUG Mode
^^^^^^^^^^^^^^
In this mode, users can further observe the intermediate results and control the internal arguments from RecWizard modules. We suggest this mode for:

3.1. When To Use 
"""""""""""""""""

- **Case 1**: Debugging the current RecWizard models at module level; 

- **Case 2**: Understanding or explaining how the RecWizard models work.

3.2. How To Use 
""""""""""""""""

In the DEBUG mode, we can not only enjoy the features like chatting or saving histories as the INFO mode, but also have the following additional features:

3.2.1. Module Dependency Visualization
***************************************

CRS models are usually complex in terms of the way to manage the data flow among modules. As the RecWizard is a modularized system, we can visualize the dependency between modules in the DEBUG mode using the `recwizard.Monitor` logics defined by the developers who shared this model.


If you need to change the `recwizard.Monitor` code or add new models with `recwizard.Monitor` code, please check our `developer guide <./development/overview>`_.


3.2.2. Intermediate Results Monitoring
***************************************

Similarly, we offer the feature to monitor the intermediate messages and results from the RecWizard modules. This feature is useful for debugging the RecWizard models at module level.

3.2.3. Intenal Arguments Tuning
********************************

In the DEBUG mode, we can also tune the internal arguments from the RecWizard modules. This feature is useful for understanding or explaining how the RecWizard models work better.
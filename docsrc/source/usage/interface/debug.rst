3. DEBUG Mode
^^^^^^^^^^^^^

In this mode, users can further observe the intermediate results and control the internal arguments from RecWizard modules. We suggest this mode for:

3.1. When To Use 
****************

- **Case 1**: Debugging the current RecWizard models at module level; 

- **Case 2**: Understanding or explaining how the RecWizard models work.

3.2. How To Use 
****************

In the DEBUG mode, we can not only enjoy the features like chatting or saving histories as the INFO mode, but also have the following additional features:

**3.2.1. Module Dependency Visualization**

CRS models are usually complex in terms of the way to manage the data flow among modules. As the RecWizard is a modularized system, we can visualize the dependency between modules in the DEBUG mode using the `recwizard.Monitor` logics defined by the developers who shared this model.


If you need to change the `recwizard.Monitor` code or add new models with `recwizard.Monitor` code, please check our `developer guide <./development/overview>`_.


**3.2.2. Intermediate Results Monitoring**

Similarly, we offer the feature to monitor the intermediate messages and results from the RecWizard modules. This feature is useful for debugging the RecWizard models at module level.

**3.2.3. Intenal Arguments Tuning**

In the DEBUG mode, we can also tune the internal arguments from the RecWizard modules. This feature is useful for understanding or explaining how the RecWizard models work better.
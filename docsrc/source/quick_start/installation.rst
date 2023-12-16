Installation
------------

.. note::

   Ensure you are using `python>=3.8`, then follow our instruction to install related packages. Note that we test RecWizard toolkit on Linux, and leave Windows / MacOS development in the future.


1. PyTorch Installation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh
   
   pip install torch # require torch>=2.0

2. Huggingface Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh
   
   pip install transformers datasets evaluate

3. RecWizard Installation
^^^^^^^^^^^^^^^^^^^^^^^^^

- RecWizard Installation via `PyPI` (Recommended)

.. code-block:: sh
   
   pip install recwizard


- RecWizard Installation Locally

.. code-block:: sh
   
   git clone git@github.com:McAuley-Lab/RecWizard.git
   cd RecWizard; pip install -e .
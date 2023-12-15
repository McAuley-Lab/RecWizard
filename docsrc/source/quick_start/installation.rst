Installation
-------------

.. note::

   Ensure you are using `python>=3.8`, then follow our instruction to install related packages. Note that we test RecWizard toolkit on Linux, and leave Windows / MacOS development in the future.


**PyTorch Installation:**

.. code-block:: sh
   
   pip install torch # require torch>=2.0

**Huggingface Installation:**

.. code-block:: sh
   
   pip install transformers datasets evaluate

**RecWizard Installation (Remotely):**

.. code-block:: sh
   
   pip install recwizard

**RecWizard Installation (Locally):**

.. code-block:: sh
   
   git clone git@github.com:McAuley-Lab/RecWizard.git
   cd RecWizard; pip install -e .
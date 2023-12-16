.. RecWizard documentation master file, created by
   sphinx-quickstart on Mon Sep 25 23:48:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome!
--------

**RecWizard** is a comprehensive toolkit designed for Conversational Recommender System (CRS) research, based on `PyTorch <https://pytorch.org>`_ and `Huggingface <https://huggingface.co/>`_.



.. grid:: 2

    .. _cards-clickable:

    .. grid-item-card:: Getting started
        :link: quick_start/concept.html

        Learn the basics and become familiar with launching and interacting with a RecWizard CRS model.


    .. grid-item-card:: User guide
        :link: usage/index.html

        Learn how to use RecWizard to build your own CRS model, and contribute to RecWizard.

.. grid:: 2

    .. _cards-clickable:

    .. grid-item-card:: Resources

        High-level overviews of the existing CRS datasets and trained models for RecWizard.

    .. grid-item-card:: API reference
        :link: reference/index.html

        Thechnical descriptions of how RecWizard classes and methods work.



.. include:: quick_start/installation.rst


.. toctree::
   :caption: GET STARTED
   :maxdepth: 2
   :hidden:

   quick_start/concept
   quick_start/installation
   quick_start/example

.. toctree::
   :caption: USER GUIDE
   :maxdepth: 2
   :hidden:

   usage/loading/index 
   usage/interface/index
   usage/development/index


.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   :hidden:

   reference/baseclass/index 
   reference/pipelines/index
   reference/modules/index




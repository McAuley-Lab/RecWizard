## Overview

This is an overview about How to create new `RecBOT` modules (such as generators, recommenders and processors) or models (such as UniCRS models).
In this tutorial, we aim to create and share a new model, called, NEW trained on [`INSPIRED`]() to go through the journey of contributing to `RecBOT`.

### 1. Tutorial Setup

We create and share the NEW as a new conversational recommender in three steps:

1. Create and train a NEW recommender;
2. Create and train a NEW generator;
3. Assemble the NEW recommender and generator under the NEW model;
4. Add interface features to NEW model;
5. More details about how to contribute to RecBOT codebase.

Let us look at the desired NEW model in advance, and check the detailed implementation steps in the corresponding sections:

#### 1.1 NEW Recommender [[details]](./module)

A NEW recommender can be used individually in  this way:

1. Raw text input and output
    ```python

    ```

2. Tensor input and output
    ```python 

    ```

#### 1.2 NEW Generator [[details]](./module)

A NEW recommender can be used individually in this way:

1. Raw text input and output
    ```python

    ```

2. Tensor input and output
    ```python 

    ```

#### 1.3 NEW Model [[details]](./model)

NEW Model can define the logitics of how to use NEW recommender and generator.

```python 

```

#### 1.4 NEW Interactive Interface [[details]](./model)

We can use `recwizard.Monitor` to define and launch the interactive interface after building the model.

```python 

```
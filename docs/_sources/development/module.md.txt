## New Modules

We are using NEW recommender and generator modules as examples to show the tour of adding new modules to RecBOT.

### 1. NEW Recommender 

Similar to standard `Huggingface` models, we need three elements for `RecBOT` modules.

#### 1.1. Define the recommender

1. Define `Config` from `recwizard.BaseConfig`:

```python 

```

2. Define `Tokenizer` from `recwizard.BaseTokenizer`

```python 

```

3. Define `Module` from `recwizard.BaseModule`

```python 

```

Nice, we can initialize our NEW recommender now!

```python 

```

#### 1.2. Train the recommender 

We follow the `Huggingface Trainer` to train the NEW recommender:

```python 

```

Also, we leave the flexiblity for the developers to train the models with any favorite strategies.

1. `Huggingface` Training Script
2. `Lightning` Training Script
3. Native `PyTorch` Training Script

#### 1.3. Save the recommender 

Now, we can tested our NEW recommender now, and share this as a pretrained model.

1. Test our trained NEW recommender 

```python 

```

2. Save our trained NEW recommender locally

```python 

```

3. Save our trained NEW recommender remotely 

```python 

```

4. Check we can use it:

```python 

```


NOTE: All the traditional (non conversational) generators can be created and shared in this way. Welcome to contribute traditional recommenders under this category!


### 2. NEW Generator 

Similar to standard `Huggingface` models, we need three elements for `RecBOT` modules.

#### 2.1. Define the generator

1. Define `Config` from `recwizard.BaseConfig`:

```python 

```

2. Define `Tokenizer` from `recwizard.BaseTokenizer`

```python 

```

3. Define `Module` from `recwizard.BaseModule`

```python 

```

Nice, we can initialize our NEW generators now!

```python 

```

#### 2.2. Train the generator 

We follow the `Huggingface Trainer` to train the NEW generator:

```python 

```

Also, we leave the flexiblity for the developers to train the generators with any favorite strategies.

1. `Huggingface` Training Script
2. `Lightning` Training Script
3. Native `PyTorch` Training Script

#### 2.3. Save the generator 

Now, we can tested our NEW generator now, and share this as a pretrained model.

1. Test our trained NEW generator 

```python 

```

2. Save our trained NEW generator locally

```python 

```

3. Save our trained NEW generator remotely 

```python 

```

4. Check we can use it:

```python 

```
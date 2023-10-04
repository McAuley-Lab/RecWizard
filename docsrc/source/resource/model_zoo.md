## Model Zoo

Please contact [Zhankui He](zhh004@ucsd.edu) if you are willing to share your `RecBOT Model` in our model zoo.

### 1. Model Types

| Name               | Model Card              | Details                                                                                                                                                                                                   | 
|--------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Switching-Decoding | [SwitchDecodeModel](./) | A Toolformer-like manager to call **recommender** during generation when encountering some special tokens.                                                                                                |
| Fill-Blank         | [FillBlankModel](./)    | A UniCRS-like manager to call **generator** first, then use **generator** outupts as part of **recommender** inputs, then use **recommender** outputs to fill into the blank in generated text.           |
| Expansion          | [ExpansionModel](./)    | A CBART-like manager to call **recommender** first, then use **recommender** outupts as part of **generator** inputs, then expand the natural language outputs surrounding the **recommender** item list. |
| LLMs               | [LLMModel](./)          | An LLM-like manager to put recommender and generator outputs into the prompt of an LLM, then use the outputs from LLMs                                                                                    |

### 2. Model Resources

#### 2.1 LLMs (`API`) as Models

| Name   | Model Card                       | Details | Domain  |
|--------|----------------------------------|---------|---------|
| OpenAI | [GPT-4](./), [GPT-3.5-turbo](./) |         | General |
| Claude | [Link](./)                       |         | General |
| Bard   | [Link](./)                       |         | General |

#### 2.2 LLMs (`Local`) as Models

| Name        | Model Card          | Details | Domain  |
|-------------|---------------------|---------|---------|
| Falcon-Chat | [7B](./), [40B](./) |         | General |
| MPT         | [7B](./), [30B](./) |         | General |
| Vicuna      | [7B](./), [13B](./) |         | General |
| BAIZE       | [7B](./), [13B](./) |         | General |

#### 2.3 Models w/ Traditional RecSys

| Name   | Model Card (By Data)                       | Details | Domain |
|--------|--------------------------------------------|---------|--------|
| ReDIAL | [ReDIAL](./), [INSPIRED](./), [Reddit](./) |         | Movie  |
| KBRD   | [ReDIAL](./), [INSPIRED](./), [Reddit](./) |         | Movie  |
| KGSF   | [ReDIAL](./), [INSPIRED](./), [Reddit](./) |         | Movie  |
| UniCRS | [ReDIAL](./), [INSPIRED](./), [Reddit](./) |         | Movie  |
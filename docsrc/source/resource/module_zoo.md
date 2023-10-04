## Module Zoo

### 1. Recommender 

#### 1.1 ID-based Recommender 

| Name     | Module Card                  | Details                                                          | 
|----------|------------------------------|------------------------------------------------------------------|
| AutoRec  | [ReDIAL](./), [INSPIRED](./) | Autoencoder-based recommender trained on ReDIAL or INSPIRED.     |
| MF       | [ReDIAL](./), [INSPIRED](./) | Matrix Factorization recommmender trained on ReDIAL or INSPIRED. |
| SASRec   | [ReDIAL](./), [INSPIRED](./) | Sequential recommmender trained on ReDIAL or INSPIRED.           |
| LightGCN | [ReDIAL](./), [INSPIRED](./) | Graph-based recommmender trained on ReDIAL or INSPIRED.          |

#### 1.2. KG-enriched Recommender 

| Name     | Module Card                  | Details                                                            | 
|----------|------------------------------|--------------------------------------------------------------------|
| KBRD-Rec | [ReDIAL](./), [INSPIRED](./) | KG-based recommender trained on ReDIAL or INSPIRED in model KBRD.  |
| KGSF-Rec | [ReDIAL](./), [INSPIRED](./) | KG-based recommmender trained on ReDIAL or INSPIRED in model KGSF. |

#### 1.3. Text-enriched Recommender 

| Name     | Module Card                  | Details                                                                  | 
|----------|------------------------------|--------------------------------------------------------------------------|
| BERT     | [ReDIAL](./), [INSPIRED](./) | BERT-based recommender trained on ReDIAL or INSPIRED in model KBRD.      |
| DialoGPT | [ReDIAL](./), [INSPIRED](./) | DialoGPT-based recommmender trained on ReDIAL or INSPIRED in model KGSF. |

#### 1.4. KG-Text-enriched Recommender 

| Name       | Module Card                  | Details                                                                                               | 
|------------|------------------------------|-------------------------------------------------------------------------------------------------------|
| UniCRS-Rec | [ReDIAL](./), [INSPIRED](./) | DialoGPT-based recommender using input text, KG entities trained on ReDIAL or INSPIRED in model KBRD. |

### 2. Generator 

#### 2.1. Template-Based Model

| Name | Module Card                    | Details                                           | 
|------|--------------------------------|---------------------------------------------------|
| MCR  | [ItemMCR](./), [BundleMCR](./) | MCR-based template rules for language generation. |

#### 2.2. Language Models

| Name     | Module Card     | Details                                                   | 
|----------|-----------------|-----------------------------------------------------------|
| RNN      | [ReDIALGRU](./) | GRU-based model trained on ReDIAL for swtiching decoding. |
| DialoGPT | [UniCRS](./)    | DialoGPT-based model trained on ReDIAL for filling blank. |

#### 2.3. Large Language Models 

| Name | Module Card                     | Details | 
|------|---------------------------------|---------|
| GPT  | [GPT3.5-turbo](./), [GPT-4](./) | .       |
| MPT  | [MPT-7B](./), [MPT-30B](./)     | .       |


### 3. Processor 

#### 3.1. Named Entity Recognition

| Name | Module Card | Details | 
| --- | --- | --- |
| FLAN-T5-NER | [Reddit-Movie](./) | A movie name extraction tool trained on Reddit-Movie dataset with FLAN-T5|
| RoBERTa-NER | [Reddit-Movie](./), [Reddit-Game](./), [Reddit-Book](./) | A named entity extraction tool trained on Reddit-Movie dataset with ReBERTa|


#### 3.2. Sentiment Analysis

| Name | Module Card | Details | 
| --- | --- | --- |
| HRNNForClassification | [ReDIAL](./) | A sentiment classifer trained on ReDIAL dataset with HRNN |


#### 3.3 Entity Linking

| Name | Module Card | Details | 
| --- | --- | --- |
| DBPedia-Linker | [DBPedia-API](./) | An API-based model to link DBPedia entities |
| WordNet-Linker | [WordNet-API](./) | An API-based model to link WordNet entities |


#### 3.4. General Processor

| Name | Module Card | Details | 
| --- | --- | --- |
| LLMs | [GPT3.5-turbo-API](./), [GPT4-API](./) | An API-based model to conduct any processing tasks with reasonable results |

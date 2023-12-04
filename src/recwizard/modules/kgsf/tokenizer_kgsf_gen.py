import json
import pickle as pkl
import numpy as np
import torch
from nltk import word_tokenize
import re
from typing import Dict, List, Callable, Union
from transformers import AutoTokenizer, BatchEncoding

from recwizard.utility.utils import WrapSingleInput, loadJsonFileFromDataset
from recwizard.tokenizer_utils import BaseTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class KGSFGenTokenizer(BaseTokenizer):
    def __init__(
            self,
            max_count: int = 5,
            max_c_length: int = 256,
            max_r_length: int = 30,
            n_entity: int = 64368,
            batch_size: int = 1,
            padding_idx: int = 0,
            entity2entityId: Dict[str, int] = None,
            word2index: Dict[str, int] = None,
            key2index: Dict[str,int] = None,
            entity_ids: List = None,
            id2name: Dict[str,str] = None,
            id2entity: Dict[int,str] = None,
            entity2id: Dict[str, int] = None,
            **kwargs,
    ):
        if entity2entityId is None:
            self.entity2entityId=pkl.load(open('../kgsfdata/entity2entityId.pkl','rb'))
        else:
            self.entity2entityId = entity2entityId
        self.entity_max=len(self.entity2entityId)
        
        if word2index is None:
            self.word2index = json.load(open('../kgsfdata/word2index_redial.json', encoding='utf-8'))
        else:
            self.word2index = word2index
            
        if key2index is None:
            self.key2index=json.load(open('../kgsfdata/key2index_3rd.json',encoding='utf-8'))
        else:
            self.key2index = key2index
            
        if entity_ids is None:
            self.entity_ids = pkl.load(open("../kgsfdata/movie_ids.pkl", "rb"))
        else:
            self.entity_ids = entity_ids

        if id2entity is None:
            self.id2entity = pkl.load(open('../kgsfdata/id2entity.pkl','rb'))
        else:
            self.id2entity = id2entity
        
        if id2name is None:
            self.id2name = json.load(open('../kgsfdata/id2name.jsonl', encoding='utf-8'))
        else:
            self.id2name = id2name
        
        if entity2id is None:
            self.entity2id = {v:k for k,v in self.id2entity.items()}
        else:
            self.entity2id = entity2id
            
        self.entityId2entity = {v:k for k,v in self.entity2entityId.items()}
        # in config:
        self.max_count = max_count
        self.max_c_length = max_c_length
        self.max_r_length = max_r_length
        self.n_entity = n_entity
        self.batch_size = batch_size
        self.pad_entity_id = padding_idx
        self.names2id = {v:k for k,v in self.id2name.items()}
        self.index2word = {v:k for k,v in self.word2index.items()}
        super().__init__(entity2id=self.entity2id, pad_entity_id=self.pad_entity_id, **kwargs)
        
    def get_init_kwargs(self):
        """
        The kwargs for initialization. They will be saved when you save the tokenizer or push it to huggingface model hub.
        """
        return {
            'entity2entityId': self.entity2entityId,
            'word2index': self.word2index,
            'key2index': self.key2index,
            'entity_ids': self.entity_ids,
            'id2entity': self.id2entity,
            'id2name': self.id2name,
        }
        
    def padding_w2v(self,sentence,max_length,pad=0,end=2,unk=3):
        """
        Pad and process a list of words into a structured representation for word vectors.
        - Converts words to numerical indices based on the provided vocabulary.
        - Calculates concept masks based on the word case.
        - Identifies DBpedia masks for entity recognition.

        Args:
            sentence (list): A list of words in the input sentence.
            max_length (int): The maximum length of the resulting sequence.
            pad (int): The padding value to use. Defaults to 0.
            end (int): The end-of-sequence value. Defaults to 2.
            unk (int): The unknown word value. Defaults to 3.

        Returns:
            Tuple[list, int, list, list]: A tuple containing four elements:
                - A list of padded and processed word indices.
                - The length of the resulting sequence.
                - A list representing concept masks.
                - A list representing DBpedia masks.

        Example:
            >>> sentence = ['Okay', ',', 'have', 'you', 'seen', '@136983', '?']
            >>> max_length = 30
            >>> padded_vector, length, concept_masks, dbpedia_masks = tokenizer.padding_w2v(sentence, max_length)
            >>> print(padded_vector)

            >>> print(length)

            >>> print(concept_masks)

            >>> print(dbpedia_masks)


        """
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        for word in sentence:
            vector.append(self.word2index.get(word,unk))
            concept_mask.append(self.key2index.get(word.lower(),0))
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id=self.entity2entityId[entity]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        if len(vector)>max_length:
            return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                    concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]
        
    def padding_context(self,contexts,pad=0):
        """
        Pad and process a list of conversation contexts into a structured representation.

        This method takes a list of conversation contexts, where each context is represented as a list of words,
        and processes them to create a structured representation suitable for model input. It pads the contexts,
        combines them into a single sequence, and calculates concept masks and DBpedia masks.

        Args:
            contexts (list of lists): Prior user interactions, each represented as a list of words.
            pad (int, optional): The padding value to use. Defaults to 0.

        Returns:
            Tuple[list, list, list]: A tuple containing three lists:
                - A list of padded and combined context sequences.
                - A list representing concept masks.
                - A list representing DBpedia masks.

        Example:
            >>> conversation_contexts = [
                ...     ['Hello'],
                ...     ['hi', 'how', 'are', 'u'],
                ...     ['Great', '.', 'How', 'are', 'you', 'this', 'morning', '?'],
                ...     ['would', 'u', 'have', 'any', 'recommendations', 'for', 'me', 'im', 'good', 'thanks', 'fo', 'asking'],
                ...     ['What', 'type', 'of', 'movie', 'are', 'you', 'looking', 'for', '?'],
                ...     ['comedies', 'i', 'like', 'kristin', 'wigg'],
                ...     ['Okay', ',', 'have', 'you', 'seen', '@136983', '?'],
                ...     ['something', 'like', 'yes', 'have', 'watched', '@140066', '?']
                ... ]

        """
        contexts_com=[]
        for sen in contexts[-self.max_count:-1]: # get the most recent max_count of contexts
            contexts_com.extend(sen)
            contexts_com.append('_split_')
        contexts_com.extend(contexts[-1])
        vec,v_l,concept_mask,dbpedia_mask=self.padding_w2v(contexts_com,self.max_c_length)
        return vec, concept_mask, dbpedia_mask

    def _names_to_id(self, input_name):
        """
        Attempt to map a movie name to its corresponding ID.

        This method takes an input movie name, processes it by stripping leading and trailing spaces,
        converting it to lowercase, and removing any year information (e.g., '(2005)'). It then compares
        the processed input name to a dictionary of movie names (in lowercase) and their corresponding IDs.
        If a match is found, the method returns the corresponding movie ID; otherwise, it returns None.

        Args:
            input_name (str): The movie name to be mapped to an ID.

        Returns:
            int or None: The corresponding movie ID if found, or None if no match is found.

        Example:
            >>> input_movie_name = "Avatar (2009)"
            >>> _names_to_id(input_movie_name)
            136983
        """
        processed_input_name = input_name.strip().lower()
        processed_input_name = re.sub(r'\(\d{4}\)', '', processed_input_name)
        
        for name, id in self.names2id.items():
            processed_name = name.strip().lower()
            if processed_input_name in processed_name:
                return id
        return None
    
    def detect_movie(self, sentence):
        """
        Detect and extract movies from a given sentence.

        This function identifies movies within a sentence by searching for text enclosed in <entity> tags
        or by using word boundaries and common punctuation. It then replaces the movie names with corresponding
        movie IDs, and if available, maps these IDs to entity IDs. The detected tokens are returned as a list,
        and any mapped entity IDs are also returned as a separate list.

        Args:
            sentence (str): The input sentence in which movies are to be detected.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing two lists:
                - A list of detected tokens in the sentence, where movie names are replaced with '@movie_id'.
                - A list of entity IDs corresponding to the detected movies.

        Example:
            >>> "I watched <entity>Avatar</entity> yesterday."
            (['I', 'watched', '@1', 'yesterday', '.'], [42])
        """
        # This regular expression pattern will match text surrounded by <movie> tags
        pattern = r"<entity>.*?</entity>|\w+|[.,!?;]"
        tokens = re.findall(pattern, sentence)
        
        # Replace movie names with corresponding IDs in the tokens
        movie_rec_trans = []
        for i, token in enumerate(tokens):
            if token.startswith('<entity>') and token.endswith('</entity>'):
                movie_name = token[len('<entity>'):-len('</entity>')]
                movie_id = self._names_to_id(movie_name)
                if movie_id is not None:
                    tokens[i] = f"@{movie_id}"
                    try:
                        entity = self.id2entity[int(movie_id)]
                        entity_id = self.entity2entityId[entity]
                        movie_rec_trans.append(entity_id)
                    except:
                        pass
        return [tokens], movie_rec_trans
    
    def encode(self,user_input=None, user_context=None, entity=None, system_response=None, movie=0):
        """
        Args:
            user_input (str): User's current input
            user_context (list of lists): Prior user interactions, each represented as a list of words.
            entity (list): Movies identified in the user context. Defaults to an empty list.
            system_response (list): The system's previous response as a list of words.
            movie (int): Identifier for movies in system_response. Defaults to 0.

        Returns:
            dict: A dictionary with the following keys and values:
                - 'context': A tensor representing the context.
                - 'response': A tensor representing the response (or None if 'system_response' is None).
                - 'concept_mask': A tensor representing concept masks.
                - 'seed_sets': A list containing 'entity'.
                - 'entity_vector': A tensor representing the entity vector.
        """
        if user_context is None:
            user_context, entity = self.detect_movie(user_input)

        print(user_context,type(user_context))
        
        entity = entity[::-1]

        context, concept_mask,dbpedia_mask=self.padding_context(user_context)
        entity_vector = np.zeros(50, dtype=int)
        point = 0
        for en in entity:
            entity_vector[point] = en
            point += 1
        context = torch.tensor([context]).to(device)
        entity = [entity]
        entity_vector = torch.tensor([entity_vector]).to(device)
        concept_mask = torch.tensor([concept_mask]).to(device)
        #context, response, seed_sets, entity_vector
        if system_response is not None:
            response,r_length,_,_=self.padding_w2v(system_response,self.max_r_length)
            response = torch.tensor([response]).to(device)
            return {
                'context':context,
                'response':response,
                'concept_mask': concept_mask,
                'seed_sets': entity,
                'entity_vector': entity_vector,
            }
            
        return {
            'context':context,
            'response':None,
            'concept_mask': concept_mask,
            'seed_sets': entity,
            'entity_vector': entity_vector,
        }
            
    def decode(self, outputs, labels=None):
        """
        Decode a batch of model outputs into human-readable sentences.

        Args:
            outputs (torch.Tensor): A tensor containing the model's output.
            labels (Optional[torch.Tensor]): An optional tensor containing the target labels, if available.

        Returns:
            List[str]: A list of decoded sentences where each sentence is a human-readable string.

        Example:
            >>> model_outputs = torch.tensor([[1, 4, 2, 3], [5, 6, 7, 2]])
            >>> tokenizer.decode(model_outputs)
            ['I watched _UNK_ .', 'Have you seen <entity>Avatar</entity> ?']
        """
        sentences=[]
        for sen in outputs.tolist():
            sentence=[]
            for word_index in sen:
                if word_index>3:
                    word = self.index2word[word_index]
                    if word[0] == '@':
                        try:
                            movie = self.id2name[word[1:]]
                            sentence.append(movie)
                            continue
                        except:
                            pass
                    sentence.append(word)
                elif word_index==3:
                    sentence.append('_UNK_')
            sentences.append(' '.join(sentence))
        return sentences

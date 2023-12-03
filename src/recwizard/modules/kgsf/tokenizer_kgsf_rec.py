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
from .utils import seed_everything
seed_everything(42)
class KGSFRecTokenizer(BaseTokenizer):

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
            self.entity2entityId=pkl.load(open('recwizard/modules/kgsf/data/entity2entityId.pkl','rb'))
        else:
            self.entity2entityId = entity2entityId
        self.entity_max=len(self.entity2entityId)
        
        if word2index is None:
            self.word2index = json.load(open('recwizard/modules/kgsf/data/word2index_redial.json', encoding='utf-8'))
        else:
            self.word2index = word2index
            
        if key2index is None:
            self.key2index=json.load(open('recwizard/modules/kgsf/data/key2index_3rd.json',encoding='utf-8'))
        else:
            self.key2index = key2index
            
        if entity_ids is None:
            self.entity_ids = pkl.load(open("recwizard/modules/kgsf/data/movie_ids.pkl", "rb"))
        else:
            self.entity_ids = entity_ids

        if id2entity is None:
            self.id2entity = pkl.load(open('recwizard/modules/kgsf/data/id2entity.pkl','rb'))
        else:
            self.id2entity = id2entity
        self.id2entity = {int(k):str(v) for k,v in self.id2entity.items()}
        if id2name is None:
            self.id2name = json.load(open('recwizard/modules/kgsf/data/id2name.jsonl', encoding='utf-8'))
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
        sentence: ['Okay', ',', 'have', 'you', 'seen', '@136983', '?'] / [...]
        max_length: 30 / 256
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
        contexts: eg. [['Hello'], ['hi', 'how', 'are', 'u'], ['Great', '.', 'How', 'are', 'you', 'this', 'morning', '?'], ['would', 'u', 'have', 'any', 'recommendations', 'for', 'me', 'im', 'good', 'thanks', 'fo', 'asking'], ['What', 'type', 'of', 'movie', 'are', 'you', 'looking', 'for', '?'], ['comedies', 'i', 'like', 'kristin', 'wigg'], ['Okay', ',', 'have', 'you', 'seen', '@136983', '?'], ['something', 'like', 'yes', 'have', 'watched', '@140066', '?']]
        """
        contexts_com=[]
        for sen in contexts[-self.max_count:-1]: # get the most recent max_count of contexts
            contexts_com.extend(sen)
            contexts_com.append('_split_')
        contexts_com.extend(contexts[-1])
        vec,v_l,concept_mask,dbpedia_mask=self.padding_w2v(contexts_com,self.max_c_length)
        return concept_mask,dbpedia_mask

    def _names_to_id(self, input_name):

        processed_input_name = input_name.strip().lower()
        processed_input_name = re.sub(r'\(\d{4}\)', '', processed_input_name)
        
        for name, id in self.names2id.items():
            processed_name = name.strip().lower()
            if processed_input_name in processed_name:
                return id
        return None
    
    def detect_movie(self, sentence):
        # This regular expression pattern will match text surrounded by <movie> tags
        pattern = r"<entity>.*?</entity>|\w+|[.,!?;]"
        print(sentence)
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
        return tokens, movie_rec_trans
    
    def encode(self,user_input, user_context=None, entity=None, system_response=None, movie=0):
        """
        user_input: eg. Hi, can you recommend a movie for me?
        user_context: eg. [['Hello'], ['hi', 'how', 'are', 'u']] TODO: 考虑分隔符吗 _split_？
        entity: movies in user_context, default []
        system_response: eg. ['Great', '.', 'How', 'are', 'you', 'this', 'morning', '?']
        movie: movies in system_response, defualt is an ID, so None. ?？？ TODO: 多个movie的话 case会重复 tokenizer怎么解决？
        """
        if user_context is None:
            user_context, entity = self.detect_movie(user_input)
        print(user_context,entity)
        user_context = [user_context]
        entity = entity[::-1]
        
        entity = torch.tensor(np.array(entity)).unsqueeze(0)
        concept_mask,dbpedia_mask=self.padding_context(user_context)
        db_vec = np.zeros(self.n_entity)
        for db in dbpedia_mask:
            if db != 0:
                db_vec[db] = 1
        concept_mask = torch.tensor(np.array(concept_mask)).unsqueeze(0)
        db_vec = torch.tensor(np.array(db_vec)).unsqueeze(0)
        if system_response is not None:
            response,r_length,_,_=self.padding_w2v(system_response,self.max_r_length)
            response = torch.tensor(np.array(response)).unsqueeze(0)
            return {
                'response':response,
                'concept_mask': concept_mask,
                'seed_sets': entity,
                'labels': movie,
                'db_vec': db_vec,
                'rec': int(movie!=0),
            }
            
        return {
            'response':None,
            'concept_mask': concept_mask,
            'seed_sets': entity,
            'labels': None,
            'db_vec': db_vec,
            'rec': 0,
        }

        
    def decode(self, outputs, top_k=3, labels=None):
        outputs = outputs[:, torch.LongTensor(self.entity_ids)] #previous outputs.shape=(1,64368), now 6924, same as len(entity_ids)
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        movieIds = []
        movieNames = []
        for i in range(top_k):
            top_pred = pred_idx[0][i]
            entity_id = int(self.entity_ids[top_pred])
            entity = self.entityId2entity[entity_id]
            movie_id = self.entity2id[entity]
            movie_name = self.id2name[str(movie_id)]
            movieIds.append(movie_id)
            movieNames.append(movie_name)
        return movieIds, movieNames
if __name__ == "__main__":
    tokenizer = KGSFRecTokenizer()
    # test_sentence = 'She also voiced a character in @131869 and @161259'
    # test_sentence = test_sentence.split(' ')
    test_sentence = 'She also voiced a character in <entity>Despicable Me 3 (2017)</entity> and <entity>How to Train Your Dragon </entity> '
    print(tokenizer.encode(test_sentence))
    # test_sentence_user = ['Hello', '_split_', 'hi', 'how', 'are', 'u', '_split_', 'Great', '.', 'How', 'are', 'you', 'this', 'morning', '?', '_split_', 'would', 'u', 'have', 'any', 'recommendations', 'for', 'me', 'im', 'good', 'thanks', 'fo', 'asking']
    # vector, ml, concept_mask, db_mask = tokenizer.padding_w2v(test_sentence,30)
    # print(vector,ml,concept_mask,db_mask)

    test_conv = [['Hello'], ['hi', 'how', 'are', 'u'], ['Great', '.', 'How', 'are', 'you', 'this', 'morning', '?'], ['would', 'u', 'have', 'any', 'recommendations', 'for', 'me', 'im', 'good', 'thanks', 'fo', 'asking'], ['What', 'type', 'of', 'movie', 'are', 'you', 'looking', 'for', '?'], ['comedies', 'i', 'like', 'kristin', 'wigg'], ['Okay', ',', 'have', 'you', 'seen', '@136983', '?'], ['something', 'like', 'yes', 'have', 'watched', '@140066', '?']]
    #concept_mask,dbpedia_mask = tokenizer.padding_context(test_conv)
    # print(vec,vl,concept_mask,dbpedia_mask,length)
    result = tokenizer.encode(test_conv,[28207, 5771],test_sentence,22727)
    print(result)
    print('passed')
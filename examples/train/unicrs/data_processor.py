import os
import re
from typing import List

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download

import logging, json

from recwizard.modules.unicrs import UnicrsGenTokenizer
from recwizard.utility.utils import loadJsonFileFromDataset
from recwizard.utility import SEP_TOKEN, EOS_TOKEN, BOS_TOKEN, HF_ORG, ENTITY_PATTERN, ENTITY_TEMPLATE



class KGDataLoader:
    debug = True
    @classmethod
    def get_entity_kg_info(cls, dataset):
        dataset_dir = os.path.join('dataset', dataset)
        if not os.path.exists(dataset_dir):
            snapshot_download(f"{HF_ORG}/{dataset}", repo_type="dataset", local_dir=dataset_dir)
        with open(os.path.join(dataset_dir, 'dbpedia_subkg.json'), 'r',
                  encoding='utf-8') as f:  # extracted in extract_subkg
            entity_kg = json.load(f)
        with open(os.path.join(dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:  # only need len
            entity2id = json.load(f)
        with open(os.path.join(dataset_dir, 'relation2id.json'), 'r',
                  encoding='utf-8') as f:  # just use len(relation2id) = len(relation_set)+1
            relation2id = json.load(f)
        with open(os.path.join(dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:  # extracted
            item_ids = json.load(f)

        edge_list = set()  # [(entity, entity, relation)]
        for entity in entity2id.values():
            if str(entity) not in entity_kg:
                continue
            for relation_and_tail in entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        edge_index = edge[:, :2].t()
        edge_type = edge[:, 2]
        num_relations = len(relation2id)
        pad_entity_id = max(entity2id.values()) + 1
        num_entities = max(entity2id.values()) + 2

        if cls.debug:
            logging.debug(
                f'#edge: {len(edge)}, #relation: {num_relations}, '
                f'#entity: {num_entities}, #item: {len(item_ids)}'
            )
        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_entities': num_entities,
            'num_relations': num_relations,
            'pad_entity_id': pad_entity_id,
            'item_ids': item_ids,
            # 'entity2id': entity2id
        }


class UnicrsDataProcessor:
    movie_pattern = re.compile(r'@(\d+)')
    entity_pattern = re.compile(ENTITY_PATTERN)
    entity_template = ENTITY_TEMPLATE
    default_movie_entity = '<movie>'
    resp_prompt = 'System:'

    def __init__(
            self,
            dataset,
            debug=False,
    ):

        super().__init__()
        self.debug = debug
        entityName2id = loadJsonFileFromDataset(dataset, 'entityName2id.json')
        self.entity2id = entityName2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}

    def mask_movie(self, text: str, movie_names: List[str]):
        for movie in sorted(movie_names, key=len, reverse=True):
            text = text.replace(movie, self.default_movie_entity)
        return text

    def mask_entities(self, text: str):
        return self.entity_pattern.sub(lambda m: m.group(1), text)


    def prepare_data_for_pretrain(self, batched_dialog):
        batch = {
            'messages': [],
            'rec': []
        }
        for dialog in batched_dialog.pa_table.to_pylist():
            resp = dialog["messages"][-1]
            entities = [self.entity2id[m.group(1)] for m in self.entity_pattern.finditer(resp) if m.group(1) in self.entity2id]
            context = dialog["messages"][:-1]
            resp = self.mask_entities(resp)
            message = SEP_TOKEN.join(context + [resp])
            for rec in list(set(entities)):
                batch['messages'].append(message)
                batch['rec'].append(rec)

        return batch


    def prepare_data_for_conv(self, dialog, gen=False):
        """
        Map data for unicrs train_conv.py. Assume each resp starts with 'System:'
        """
        context = SEP_TOKEN.join(dialog["messages"][:-1]+[self.resp_prompt])
        resp = dialog["messages"][-1]
        resp = resp[resp.find(':')+1:] # remove 'System:'
        resp = self.mask_entities(self.mask_movie(resp, dialog['recNames']))

        if gen:
            context = BOS_TOKEN + context
            resp = resp + EOS_TOKEN
        else:
            context = BOS_TOKEN + context + resp + EOS_TOKEN
            resp = ''

        return {
            'context': context,
            'resp': resp
        }

    def prepare_data_for_rec(self, batched_dialog, indices, gen_data=None, use_resp=False):
        """
        Map data for unicrs train_rec.py. Assume each resp starts with 'System:'
        """
        batch = {
            'messages': [],
            'rec': []
        }
        dialogs = batched_dialog.pa_table.to_pylist()
        for i, dialog in zip(indices, dialogs):
            # we don't use the original response, but the one from the generation
            context = SEP_TOKEN.join(dialog["messages"][:-1] + [''])
            if use_resp:
                pred = gen_data[i]['pred']
                if self.default_movie_entity in pred:
                    context += self.resp_prompt + ' ' + pred
            for rec in dialog['rec']:
                batch['messages'].append(context)
                batch['rec'].append(rec)
        return batch

    def merge_gen_data(self, dialog, idx, gen_data):
        message = dialog["messages"]
        pred = gen_data[idx]['pred']
        resp_start = message.rfind(self.resp_prompt) + len(self.resp_prompt)

        if self.default_movie_entity in pred:
            message = message[:resp_start] + ' ' + pred # The space was stripped by the evaluator, we need to add it back
        else:
            message = message[:resp_start]
        return {
            'messages': message,
            'rec': dialog['rec']
        }

if __name__ == '__main__':
    dataset = "inspired_unicrs"
    dp = UnicrsDataProcessor(dataset)
    datasets = load_dataset(os.path.join('recwizard', dataset))
    tokenizer = UnicrsGenTokenizer.from_pretrained('recwizard/UnicrsGen-inspired')

    # for subset in ["train", "validation", "test"]:
    #     datasets[subset] = datasets[subset].select(range(20))
    datasets = datasets.filter(lambda x: x['messages'][-1].startswith('System:'))
    datasets = datasets.map(dp.prepare_data_for_conv, load_from_cache_file=False)
    import collections
    global tot, word_cnt
    tot = 0
    word_cnt = collections.defaultdict(int)
    def count(dialog):
        global tot, word_cnt
        tot += 1
        context = dialog['context']
        resp_start = context.rfind('System:') + len('System: ')
        first_w = context[resp_start:].strip().split()[0]
        word_cnt[first_w] += 1
    datasets.map(count, load_from_cache_file=False)
    print(tot, {k: word_cnt[k] for k in sorted(word_cnt.keys(), key=lambda x: -word_cnt[x])[:10]})
    # tokenized_datasets = processed_datasets.map(lambda x: tokenizer(x['context']))
    # for i in range(10):
    #     data = processed_datasets['train'][i]
    #     encoded = tokenizer(data['context'])
    #     ids = encoded['context']['input_ids']
    #     decoded = tokenizer.batch_decode(ids)[0]
    #     print(encoded)
    #     print(decoded)
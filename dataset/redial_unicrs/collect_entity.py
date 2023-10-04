import json
import re
import os


def collect_entity_mappings():
    files = [
        "test_data_dbpedia.jsonl",
        "train_data_dbpedia.jsonl",
        "valid_data_dbpedia.jsonl"
        ]
    entityName2entity = {}

    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)

    for path in files:
        with open(path, 'r') as f:
            for line in f:
                dialog = json.loads(line)
                for message in dialog["messages"]:
                    for name, entity in zip(message['entity_name'], message['entity']):
                        entityName2entity[name] = entity
                    for movie_name, entity in zip(message['movie_name'], message['movie']):
                        entityName2entity[movie_name] = entity


    entityName2id = {k: entity2id[entityName2entity[k]] for k in entityName2entity}
    with open('entityName2id.json', 'w') as f:
        json.dump(entityName2id, f)


if __name__ == "__main__":
    collect_entity_mappings()
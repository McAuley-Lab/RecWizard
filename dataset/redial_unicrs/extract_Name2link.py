import json
import re

with open('./entity2id.json') as entity2id, open('./entityName2id.json') as entityName2id:
    entity2id = json.load(entity2id)
    entityName2id = json.load(entityName2id)

    id2entity = {k: v for v, k in entity2id.items()}
    print('entity2id has:', len(entity2id))
    print('id2entity has:', len(id2entity))
    print('entityName2id has:', len(entityName2id))


    entityName2link = {en : id2entity[entityName2id[en]] for en in entityName2id}
    json.dump(entityName2link, open('entityName2link.json', 'w'))

    # entityName2link = {}
    # film_pattern = re.compile(r'/([^/]*\((\d+_)?film\))>')
    # for link, id in entity2id.items():
    #     match = film_pattern.search(link)
    #     if match:
    #         name = match.group(1).replace('_', ' ')
    #         name = ' '.join(name.split(' ')).strip(' ')
    #         film_start = name.rfind('film')
    #         name = name[:film_start-1] + name[film_start+4:]
    #         if name[-2] == '(':
    #             name = name[:-2]
    #         entityName2link[name] = link
    # print('entity2link has:', len(entityName2link))
    # json.dump(entityName2link, open('entity2link.json', 'w'))


import json
import csv
import re

entityLink2id = json.load(open('entity2id.json'))

reader = csv.reader(open('movies_merged.csv'))

date_pattern = re.compile(r'\(\d+\)')

entity2link = {}

temp1 = "<http://dbpedia.org/resource/{}_(film)>"
temp2 = "<http://dbpedia.org/resource/{}_({}_film)>"
temp3 = "<http://dbpedia.org/resource/{}>"


for row in reader:
    if row[0] == 'index':
        continue
    entity = row[1].strip('"')
    match = date_pattern.search(entity)
    if match:
        movieName = entity[:match.start()].strip(' ')
        year = match.group(0)[1:-1]
    else:
        movieName = entity.strip(' ')
        year = ''
    movieName = movieName.replace(' ', '_')
    if (t1 := temp1.format(movieName)) in entityLink2id:
        entity2link[entity] = t1
    elif (t2 := temp2.format(movieName, year)) in entityLink2id:
        entity2link[entity] = t2
    elif (t3 := temp3.format(movieName)) in entityLink2id:
        entity2link[entity] = t3



print('entity2link: ', len(entity2link))
for e, link in entity2link.items():
    entity2link[e] = link[1:-1]
json.dump(entity2link, open('entity2link.json', 'w'))




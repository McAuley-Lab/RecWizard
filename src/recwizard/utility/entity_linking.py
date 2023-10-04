from .utils import loadJsonFileFromDataset
import urllib.parse

class EntityLink:
    entityName2link = None

    def __init__(self, dataset='redial', fileName='entityName2link.json'):
        self.entityName2link = loadJsonFileFromDataset(dataset, fileName, dataset_org='recwizard')

    def __call__(self, entityName):
        return self.entityName2link.get(entityName,
                                        f'https://www.google.com/search?q={urllib.parse.quote_plus(entityName)}')

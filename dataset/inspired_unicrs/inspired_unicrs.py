import json
from typing import List
import datasets
import re

ENTITY = 'entity'
ENTITY_PATTERN = r'<entity>{}</entity>'


def markup_entity(utt: str, entities: List[str]):
    # If entities like "action movie" and "action" appear at the same time, we only mark the longer one
    entities = sorted(list(set(entities)), key=lambda x: len(x), reverse=True)
    for i, entity in enumerate(entities):
        valid = entity not in ENTITY
        for prev in entities[:i]:
            if entity in prev:
                valid = False
        if valid:
            utt = re.sub(entity, ENTITY_PATTERN.format(entity), utt)
    return utt

logger = datasets.logging.get_logger(__name__)


_URL = "./"
_URLS = {
    "train": _URL + "train_data_dbpedia.jsonl",
    "valid": _URL + "valid_data_dbpedia.jsonl",
    "test": _URL + "test_data_dbpedia.jsonl",
    "entity2id": _URL + "entity2id.json"
}


class InspiredConfig(datasets.BuilderConfig):
    def __init__(self, features,
                 initiator_prefix='User: ',
                 respondent_prefix='System: ',
                 **kwargs):
        """BuilderConfig for Inspired (used in UniCRS).

        Args:
            features: *list[string]*, list of the features that will appear in the
                feature dict. Should not include "label".
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.features = features
        self.initiator_prefix = initiator_prefix
        self.respondent_prefix = respondent_prefix

class Inspired(datasets.GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "unrolled"
    BUILDER_CONFIGS = [
        InspiredConfig(
            name="unrolled",
            description="The processed Inspired dataset in UniCRS. Each conversation yields multiple samples",
            features={
                "messages": datasets.Sequence(datasets.Value("string")),
                "rec": datasets.Sequence(datasets.Value("int32")),
                "recNames": datasets.Sequence(datasets.Value("string")),
            }
        )
    ]


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(self.config.features),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        entity2id_file = downloaded_files["entity2id"]
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": downloaded_files["train"], "entity2id": entity2id_file}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": downloaded_files["valid"], "entity2id": entity2id_file}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": downloaded_files["test"], "entity2id": entity2id_file}),
        ]



    def _generate_examples(self, filepath, entity2id):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with open(entity2id, 'r', encoding='utf-8') as f:
            entity2id = json.load(f)
        if "unrolled" in self.config.name:
            Idx = 0
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    dialog = json.loads(line)
                    context = []

                    for turn in dialog:
                        resp = turn['text']
                        movie_turn = [entity2id[movie] for movie in turn['movie_link'] if movie in entity2id]

                        resp = markup_entity(resp, turn['entity_name']+turn['movie_name'])
                        prefix = self.config.initiator_prefix if turn['role'] == 'SEEKER' else self.config.respondent_prefix
                        resp = prefix + resp
                        context.append(resp)

                        yield Idx, {
                            'messages': context,
                            'rec': movie_turn,
                            'recNames': turn['movie_name']
                        }

                        Idx += 1



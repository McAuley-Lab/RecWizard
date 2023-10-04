import json
import re
from typing import List
import html
import datasets

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


class RedialConfig(datasets.BuilderConfig):
    """BuilderConfig for ReDIAL."""

    def __init__(self, features,
                 initiator_prefix='User: ',
                 respondent_prefix='System: ',
                 **kwargs):
        """BuilderConfig for ReDIAL (used in UniCRS).

        Args:
        features: *list[string]*, list of the features that will appear in the
            feature dict. Should not include "label".
        **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.features = features
        self.initiator_prefix = initiator_prefix
        self.respondent_prefix = respondent_prefix

_URL = "./"
_URLS = {
    "train": _URL + "train_data_dbpedia.jsonl",
    "valid": _URL + "valid_data_dbpedia.jsonl",
    "test": _URL + "test_data_dbpedia.jsonl",
    "entity2id": _URL + "entity2id.json"
}


class ReDIAL(datasets.GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "unrolled"
    BUILDER_CONFIGS = [

        RedialConfig(
            name="compact",
            description="Each conversation is one sample",
            features={
                "movieIds": datasets.Sequence(datasets.Value("string")),
                "movieNames": datasets.Sequence(datasets.Value("string")),
                "initiatorWorkerId": datasets.Value("int32"),
                "messages": datasets.Sequence(datasets.Value("string")),
                "senders": datasets.Sequence(datasets.Value("int32")),
                "entities": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "movies": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
            },
        ),

        RedialConfig(
            name="unrolled",
            description="Formatted unrolled dialog messages. The `rec` and `recNames` are the movies from the last message",
            features={
                "messages": datasets.Sequence(datasets.Value("string")),
                "rec": datasets.Sequence(datasets.Value("int32")),
                "recNames": datasets.Sequence(datasets.Value("string")),
            }
        ),
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

    def _process_utt(self, utt, movieid2name, replace_movieId, remove_movie=False):
        def convert(match):
            movieid = match.group(0)[1:]
            if movieid in movieid2name:
                if remove_movie:
                    return '<movie>'
                movie_name = movieid2name[movieid]
                movie_name = ' '.join(movie_name.split())
                return movie_name
            else:
                return match.group(0)

        if replace_movieId:
            utt = re.sub(self.movie_pattern, convert, utt)
        utt = ' '.join(utt.split())
        utt = html.unescape(utt)

        return utt

    movie_pattern = re.compile(r'@(\d+)')
    default_movie_entity = '<movie>'

    def _generate_examples(self, filepath, entity2id):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with open(entity2id, 'r', encoding='utf-8') as f:
            entity2id = json.load(f)
        if self.config.name == "unrolled":
            Idx = 0
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    dialog = json.loads(line)
                    if len(dialog['messages']) == 0:
                        continue

                    movieid2name = dialog['movieMentions']
                    user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
                    context, resp = [], ''
                    messages = dialog['messages']
                    turn_i = 0
                    while turn_i < len(messages):
                        worker_id = messages[turn_i]['senderWorkerId']
                        utt_turn = []
                        movie_turn = []
                        movie_turn_names = []
                        turn_j = turn_i
                        while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                            utt = self._process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True)
                            utt = markup_entity(utt, messages[turn_j]['entity_name']+messages[turn_j]['movie_name'])
                            utt_turn.append(utt)
                            movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if
                                         movie in entity2id]
                            movie_turn.extend(movie_ids)
                            movie_turn_names.extend(messages[turn_j]['movie_name'])
                            turn_j += 1

                        utt = ' '.join(utt_turn)
                        prefix = self.config.initiator_prefix if worker_id == user_id else self.config.respondent_prefix
                        resp = prefix + utt
                        context.append(resp)

                        yield Idx, {
                            'messages': context,
                            'rec': movie_turn,
                            'recNames': movie_turn_names,
                        }
                        Idx += 1
                        turn_i = turn_j

        elif self.config.name == "compact":
            Idx = 0
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    dialog = json.loads(line)
                    if len(dialog['messages']) == 0:
                        continue
                    messages = dialog['messages']

                    movieIds = [movieId for movieId in dialog["movieMentions"]]
                    yield Idx, {
                        "movieIds": movieIds,
                        "movieNames": [dialog["movieMentions"][id] for id in movieIds],
                        "initiatorWorkerId": dialog["initiatorWorkerId"],
                        "messages": [turn['text'] for turn in messages],
                        "senders": [turn["senderWorkerId"] for turn in messages],
                        "entities": [[entity2id[entity] for entity in turn['entity'] if entity in entity2id] for turn in
                                     messages],
                        "movies": [[entity2id[entity] for entity in turn['movie'] if entity in entity2id] for turn in
                                   messages]
                    }

                    Idx += 1

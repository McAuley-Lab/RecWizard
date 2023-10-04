import json
import re
from typing import List
import html
import datasets

ENTITY = 'entity'
ENTITY_PATTERN = r'<entity>{}</entity>'

logger = datasets.logging.get_logger(__name__)


class RedialConfig(datasets.BuilderConfig):
    """BuilderConfig for ReDIAL."""

    def __init__(self, features,
                 initiator_prefix='User: ',
                 respondent_prefix='System: ',
                 **kwargs):
        """BuilderConfig for ReDIAL.

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
    "train": _URL + "train.jsonl",
    "valid": _URL + "valid.jsonl",
    "test": _URL + "test.jsonl",
}


class ReDIAL(datasets.GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "rec"
    BUILDER_CONFIGS = [

        RedialConfig(
            name="SA",
            description="For using the ReDIAL dataset to train sentiment analysis on movies in sentences",
            features={
                "movieId": datasets.Value("int32"),
                "movieName": datasets.Value("string"),
                "messages": datasets.features.Sequence(datasets.Value("string")),
                "senders": datasets.features.Sequence(datasets.Value("int32")),
                "form": datasets.features.Sequence(
                    datasets.Value("int32"), length=6
                )
            },
            # certain information(e.g. movie_occurrences) is model-specific, and we leave it for Dataset.map
        ),
        # RedialConfig(
        #     name="SA_debug",
        #     description="For using the ReDIAL dataset to train sentiment analysis on movies in sentences",
        #     features={
        #         "id": datasets.Value("int32"),
        #         "movieName": datasets.Value("string"),
        #         "messages": datasets.features.Sequence(datasets.Value("string")),
        #         "senders": datasets.features.Sequence(datasets.Value("int32")),
        #         "form": datasets.features.Sequence(
        #             datasets.Value("int32"), length=6
        #         )
        #     },
        # ),
        RedialConfig(
            name="autorec",
            description="For training autorec model on ReDIAL data",
            features=datasets.Features({
                "movieIds": datasets.Sequence(datasets.Value("int32")),
                "ratings": datasets.Sequence(datasets.Value("float"))
            }),
        ),
        RedialConfig(
            name="rec",
            description="For using the ReDIAL dataset to train recommender",
            features={
                "movieIds": datasets.Sequence(datasets.Value("int32")),
                "messages": datasets.features.Sequence(datasets.Value("string")),
                "senders": datasets.features.Sequence(datasets.Value("int32")),
            },
        ),
        RedialConfig(
            name="formatted",
            description='Embed all information into a text sequence for each dialog',
            features={
                "messages": datasets.features.Sequence(datasets.Value("string")),
            }
        )
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_sender = None

    def _processMessage(self, msg, initialId):
        """
        msg example:     {
          "timeOffset": 0,
          "text": "Hi I am looking for a movie like @111776",
          "senderWorkerId": 956,
          "messageId": 204171
        },
        """
        res = {
            "text": msg["text"],
            "sender": 1 if msg["senderWorkerId"] == initialId else -1
        }
        return res

    def _flattenMessages(self, conversation, add_prefix=False):
        messages = []
        senders = []
        for message in conversation["messages"]:
            role = 1 if message["senderWorkerId"] == conversation["initiatorWorkerId"] else -1
            text = message["text"]
            if len(senders) > 0 and senders[-1] == role:
                messages[-1] += "\n" + text
            else:
                senders.append(role)
                if add_prefix:
                    prefix = self.config.initiator_prefix if role == 1 else self.config.respondent_prefix
                    text = prefix + text
                messages.append(text)
        return messages, senders

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(self.config.features),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["valid"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    movie_pattern = re.compile(r'@(\d+)')
    default_movie_entity = '<movie>'

    def _process_utt(self, utt, movieid2name, replace_movieId=True, remove_movie=False):
        def convert(match):
            movieid = match.group(0)[1:]
            if movieid in movieid2name:
                if remove_movie:
                    return '<movie>'
                movie_name = movieid2name[movieid]
                movie_name = ' '.join(movie_name.split())
                return ENTITY_PATTERN.format(movie_name)
            else:
                return match.group(0)

        if replace_movieId:
            utt = re.sub(self.movie_pattern, convert, utt)
        utt = ' '.join(utt.split())
        utt = html.unescape(utt)

        return utt

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        if self.config.name == "autorec":
            with open(filepath, encoding="utf-8") as f:
                idx = 0
                for line in f:
                    conversation = json.loads(line)
                    movieIds = []
                    ratings = []
                    if len(conversation["initiatorQuestions"]) == 0:
                        continue
                    for id, form in conversation["initiatorQuestions"].items():
                        rating = int(form["liked"])
                        if rating < 2:
                            movieIds.append(id)
                            ratings.append(rating)
                    if len(movieIds) > 0:
                        yield idx, {
                            "movieIds": movieIds,
                            "ratings": ratings
                        }
                        idx += 1

        elif "SA" in self.config.name:
            Idx = 0
            date_pattern = re.compile(r'\(\d{4}\)')  # To match e.g. "(2009)"
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    conversation = json.loads(line)
                    init_q = conversation["initiatorQuestions"]
                    resp_q = conversation["respondentQuestions"]
                    msgs, senders = self._flattenMessages(conversation)
                    # get movies that are in both forms.
                    gen = [key for key in init_q if key in resp_q]
                    for id in gen:
                        # remove date from movie name
                        movieName = date_pattern.sub('', conversation["movieMentions"][id]).strip(" ")
                        if len(movieName) == 0:
                            continue
                        yield Idx, {
                            "movieId": int(id),
                            "movieName": movieName,
                            "messages": msgs,
                            "senders": senders,
                            "form": [init_q[id]["suggested"], init_q[id]["seen"], init_q[id]["liked"],
                                     resp_q[id]["suggested"], resp_q[id]["seen"], resp_q[id]["liked"], ]
                        }
                        Idx += 1
                    if Idx > 100 and "debug" in self.config.name:
                        break
        elif "rec" in self.config.name:
            Idx = 0
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    conversation = json.loads(line)
                    msgs, senders = self._flattenMessages(conversation)

                    yield Idx, {
                        "messages": msgs,
                        "senders": senders,
                        "movieIds": [int(movieId) for movieId in conversation["movieMentions"]]
                    }
                    Idx += 1
        elif "formatted" in self.config.name:
            Idx = 0
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    dialog = json.loads(line)
                    msgs, senders = self._flattenMessages(dialog, add_prefix=True)
                    movieid2name = dialog['movieMentions']
                    formatted_msgs = [self._process_utt(utt, movieid2name) for utt in msgs]
                    yield Idx, {
                        "messages": formatted_msgs,
                    }
                    Idx += 1

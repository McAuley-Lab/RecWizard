from recwizard.configuration_utils import BaseConfig


class RedialGenConfig(BaseConfig):
    def __init__(self, hrnn_params=None, decoder_params=None, vocab_size=15005, n_movies=6924, **kwargs):
        super().__init__(**kwargs)
        self.hrnn_params = hrnn_params
        self.decoder_params = decoder_params
        self.vocab_size = vocab_size
        self.n_movies = n_movies

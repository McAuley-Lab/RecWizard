from recwizard.configuration_utils import BaseConfig


class RedialRecConfig(BaseConfig):
    def __init__(self, sa_params=None, autorec_params=None, n_movies=6924, **kwargs):
        super().__init__(**kwargs)
        self.sa_params = sa_params
        self.autorec_params = autorec_params
        self.n_movies = n_movies

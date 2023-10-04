# Model hyper parameters

hrnn_params = {
    'use_movie_occurrences': False,
    'sentence_encoder_hidden_size': 256,
    'conversation_encoder_hidden_size': 256,
    'sentence_encoder_num_layers': 1,
    'conversation_encoder_num_layers': 1,
    'use_dropout': False,
    'conv_bidirectional': False,
    'sentence_encoder_model': "princeton-nlp/unsup-simcse-roberta-base"
}

sentiment_analysis_params = {
    "return_liked_probability": False,
    "multiple_items_per_example": False,
    "output_classes": {'i_suggested': 1,
                       'i_seen': 3,
                       'i_liked': 3,
                       'r_suggested': 1,
                       'r_seen': 3,
                       'r_liked': 3},
    'hrnn_params': {
        # whether to add a dimension indicating the occurrence of the movie name in the sentence
        'use_movie_occurrences': 'word',
        'sentence_encoder_hidden_size': 512,
        'conversation_encoder_hidden_size': 512,
        'sentence_encoder_num_layers': 2,
        'conversation_encoder_num_layers': 2,
        'use_dropout': 0.4,
        'conv_bidirectional': False,
        'sentence_encoder_model': "princeton-nlp/unsup-simcse-roberta-base"
    }
}

autorec_params = {
    'layer_sizes': [1000],
    'f': "sigmoid",
    'g': "sigmoid",
}

decoder_params = {
    "context_size": 256,  # should match hrnn_params['conversation_encoder_hidden_size']
    "hidden_size": 256,
    "num_layers": 1,
    "peephole": False
}

recommender_params = {
    'decoder_params': decoder_params,
    'hrnn_params': hrnn_params,
    "sentiment_analysis_params": sentiment_analysis_params,
    "autorec_params": autorec_params,
    'latent_layer_sizes': None,
    'language_aware_recommender': False,
}

# Training parameters
train_sa_params = {
    "learning_rate": 0.001,
    "batch_size": 16,
    "nb_epochs": 50,
    "patience": 5,

    "weight_decay": 0,
    "use_class_weights": True,  # whether to use class weights to reduce class imbalance for liked? label
    "cut_dialogues": -1,  # if >=0, specifies the width of the cut around movie mentions. Otherwise, don't cut dialogues
    "targets": "suggested seen liked"
}
train_autorec_params = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "nb_epochs": 50,
    "patience": 5,

    "batch_input": "random_noise",
    "max_num_inputs": 1e10  # max number of inputs (for random_noise batch loading mode)
}
train_recommender_params = {
    "learning_rate": 0.001,
    "batch_size": 4,
    "nb_epochs": 50,
    "patience": 5,
}

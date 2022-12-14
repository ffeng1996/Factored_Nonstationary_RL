# This module implements specialized container datatypes providing alternatives
from collections import namedtuple

HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'z_size',  # latent state sizes
                                         'action_size',
                                         'reward_size',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',  # number of rnn cells
                                         'batch_size',
                                         'vae_batch_size',
                                         'vae_itr_num',
                                         'grad_clip',
                                         'num_mixture',
                                         'temperature',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                         'kl_tolerance',
                                         ])


def default_hps(game):
    return HyperParams(num_steps=10000,
                       z_size=20,
                       action_size=action_size,
                       reward_size=1,
                       max_seq_len=max_seq_len,
                       input_seq_width=21,
                       output_seq_width=20,
                       rnn_size=256,
                       batch_size=bs,
                       vae_batch_size=100,
                       vae_itr_num=10,
                       grad_clip=1,
                       num_mixture=2,
                       temperature=0.7,
                       learning_rate=lr,
                       decay_rate=dr,
                       min_learning_rate=0.00001,
                       use_layer_norm=0,
                       use_recurrent_dropout=0,
                       recurrent_dropout_prob=0.90,
                       use_input_dropout=0,
                       input_dropout_prob=0.90,
                       use_output_dropout=0,
                       output_dropout_prob=0.90,
                       is_training=is_train,
                       kl_tolerance=0.5)
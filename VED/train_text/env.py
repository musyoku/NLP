# -*- coding: utf-8 -*-
import sys, os
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from model import Conf, Model

dataset, n_vocab, n_dataset = vocab.load(args.text_dir)
conf = Conf()
conf.gpu_enabled = False if args.gpu_enabled == -1 else True
conf.n_vocab = n_vocab

# Embed
conf.char_embed_size = 20
conf.word_embed_size = 100

# Encoder
conf.word_encoder_lstm_units = [3]
conf.word_encoder_lstm_apply_batchnorm = False
conf.word_encoder_fc_hidden_units = [500, 500]
conf.word_encoder_fc_apply_batchnorm = True
conf.word_encoder_fc_apply_dropout = False
conf.word_encoder_fc_nonlinear = "elu"

# Decoder
conf.word_decoder_lstm_units = [500]
conf.word_decoder_lstm_apply_batchnorm = False
conf.word_decoder_merge_type = "concat"

# Discriminator
conf.discriminator_hidden_units = [500, 500]
conf.discriminator_apply_batchnorm = True
conf.discriminator_apply_dropout = False
conf.discriminator_nonlinear = "elu"

# Trainer
conf.learning_rate = 0.0003
conf.gradient_momentum = 0.95

model = Model(conf, name="m1")
model.load(args.model_dir)
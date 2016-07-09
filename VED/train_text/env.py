# -*- coding: utf-8 -*-
import sys, os
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from model import Conf, Model

dataset, n_vocab, n_dataset = vocab.load(args.text_dir)
conf = Conf()
conf.use_gpu = False if args.use_gpu == -1 else True
conf.n_vocab = n_vocab

# Embed
conf.char_embed_size = 20
conf.word_embed_size = 200

# Encoder
conf.word_encoder_lstm_units = [500]
conf.word_encoder_lstm_apply_batchnorm = False
conf.word_encoder_fc_hidden_units = []
conf.word_encoder_fc_apply_batchnorm = False
conf.word_encoder_fc_apply_dropout = False
conf.word_encoder_fc_nonlinear = "elu"

# Decoder
conf.word_decoder_lstm_units = [500]
conf.word_decoder_lstm_apply_batchnorm = False
conf.word_decoder_fc_hidden_units = []
conf.word_decoder_fc_apply_batchnorm = False
conf.word_decoder_fc_apply_dropout = False
conf.word_decoder_fc_nonlinear = "elu"
conf.word_decoder_merge_type = "concat"

# Trainer
conf.learning_rate = 0.0003
conf.gradient_momentum = 0.95

model = Model(conf, name="m1")
model.load(args.model_dir)
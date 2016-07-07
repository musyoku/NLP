# -*- coding: utf-8 -*-
import sys, os
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from lstm import Conf, LSTM

dataset, n_vocab, n_dataset = vocab.load(args.text_dir)
conf = Conf()
conf.use_gpu = False if args.use_gpu == -1 else True
conf.n_vocab = n_vocab
conf.lstm_apply_batchnorm = True if args.lstm_apply_batchnorm == 1 else False
conf.lstm_hidden_units = [1000]
conf.fc_output_type = args.fc_output_type
conf.fc_hidden_units = [500]
conf.fc_apply_batchnorm = False
conf.fc_apply_dropout = False
conf.forget_hidden_units = [500]
conf.forget_apply_batchnorm = False
conf.forget_apply_dropout = False
conf.embed_size = 200
lstm = LSTM(conf, name="lstm")
lstm.load(args.model_dir)
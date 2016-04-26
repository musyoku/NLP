# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config(object):
	def __init__(self):
		self.use_gpu = True
		self.learning_rate = 0.00025
		self.gradient_momentum = 0.95
		self.n_vocab = -1

		self.ndim_char_embed = 200
		self.ndim_m = 1024
		self.ndim_g = 4096

		self.bi_lstm_units = [self.ndim_char_embed, 4096]
		self.bi_lstm_apply_dropout = False

		self.attention_fc_units = [self.ndim_m, 1]
		self.attention_fc_hidden_activation_function = "elu"
		self.attention_fc_output_activation_function = None
		self.attention_fc_apply_dropout = False

		self.reader_fc_units = [self.ndim_g, 2048, self.ndim_char_embed]
		self.reader_fc_hidden_activation_function = "elu"
		self.reader_fc_output_activation_function = None
		self.reader_fc_apply_dropout = False

	def check(self):
		if len(self.bi_lstm_units) < 1:
			raise Exception("You need to add one or more hidden layers to LSTM network.")

config = Config()

# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config(object):
	def __init__(self):
		self.use_gpu = True
		self.learning_rate = 0.00025
		self.gradient_momentum = 0.95
		self.n_vocab = -1

		self.char_embed_size = 200
		self.representation_size = 400

		self.bi_lstm_units = [self.embed_size, 768]
		self.bi_lstm_apply_batchnorm = False
		self.bi_lstm_apply_batchnorm_to_input = False
		self.bi_lstm_apply_dropout = False

	def check(self):
		if len(self.bi_lstm_units) < 1:
			raise Exception("You need to add one or more hidden layers to LSTM network.")

config = Config()

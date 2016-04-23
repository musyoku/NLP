# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config(object):
	def __init__(self):
		self.use_gpu = True
		self.learning_rate = 0.00025
		self.gradient_momentum = 0.95

		self.embed_size = 100
		self.intermidiate_size = 300

		self.enc_lstm_units = [self.embed_size, 768]
		self.enc_lstm_apply_batchnorm = False
		self.enc_lstm_apply_batchnorm_to_input = False
		self.enc_lstm_apply_dropout = False

		self.itm_lstm_units = [self.enc_lstm_units[-1] + self.intermidiate_size, 1024]
		self.itm_lstm_apply_batchnorm = False
		self.itm_lstm_apply_batchnorm_to_input = False
		self.itm_lstm_apply_dropout = False

		self.dec_lstm_apply_batchnorm = False
		self.dec_lstm_apply_batchnorm_to_input = False
		self.dec_lstm_apply_dropout = False

		self.dec_fc_apply_batchnorm = False
		self.dec_fc_apply_batchnorm_to_input = False
		self.dec_fc_apply_batchnorm_to_output = False
		self.dec_fc_apply_dropout = False
		self.dec_fc_activation_function = "elu"

	@property
	def n_vocab(self):
		return self._n_vocab

	@n_vocab.setter
	def n_vocab(self, value):
		self.dec_lstm_units = [self.itm_lstm_units[-1] + value, 1024]
		self.dec_fc_units = [self.dec_lstm_units[-1], 1024]
		self._n_vocab = value

	def check(self):
		if len(self.enc_lstm_units) < 2:
			raise Exception("You need to add one or more hidden layers to LSTM network.")
		if len(self.dec_lstm_units) < 2:
			raise Exception("You need to add one or more hidden layers to LSTM network.")
		if len(self.dec_fc_units) < 1:
			raise Exception("You need to add one or more hidden layers to fully-connected network.")

config = Config()

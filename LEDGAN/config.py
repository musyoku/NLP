# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config(object):
	def __init__(self):
		self.use_gpu = True
		self.learning_rate = 0.00025
		self.gradient_momentum = 0.95

		# Generator
		## zの次元数
		self.z_size = 512

		# Discriminator
		## 文字埋め込みベクトルの次元数
		self.embed_size = 200

		# 各LSTMレイヤのユニット数を入力側から出力側に向かって並べる
		## e.g 500(input vector)->250->100(output vector)
		## self.q_fc_units = [500, 250, 100]
		self.enc_lstm_units = [self.embed_size, 1024, 1024]
		self.enc_lstm_apply_batchnorm = False
		self.enc_lstm_apply_batchnorm_to_input = False
		self.enc_lstm_apply_dropout = False

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
		# Decoderは1時刻前のy（n_vocab次元）とh（enc_lstm_units[-1]次元）を入力する
		self.dec_lstm_units = [self.enc_lstm_units[-1] + value, 1024, 1024]
		# Decoder LSTM出力をIDに変換する全結合層の各ユニット数 
		## 出力であるソフトマックス層は自動的に挿入されます
		self.dec_fc_units = [self.dec_lstm_units[-1], 2048]
		self._n_vocab = value

	def check(self):
		if len(self.enc_lstm_units) < 2:
			raise Exception("You need to add one or more hidden layers to LSTM network.")
		if len(self.dec_lstm_units) < 2:
			raise Exception("You need to add one or more hidden layers to LSTM network.")
		if len(self.dec_fc_units) < 1:
			raise Exception("You need to add one or more hidden layers to fully-connected network.")

config = Config()

# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config:
	def __init__(self):
		self.n_vocab = -1
		self.n_dataset = -1
		self.use_gpu = True
		self.learning_rate = 0.00025
		self.gradient_momentum = 0.95

		# 文字埋め込みベクトルの次元数
		self.embed_size = 200

		# 各LSTMレイヤのユニット数を入力側から出力側に向かって並べる
		## e.g 500(input vector)->250->100(output vector)
		## self.q_fc_units = [500, 250, 100]
		self.lstm_units = [self.embed_size, 512]
		self.lstm_apply_batchnorm = False
		self.lstm_apply_batchnorm_to_input = False
		self.lstm_apply_dropout = False

		# LSTM出力をIDに変換する全結合層の各ユニット数 
		## 出力であるソフトマックス層は自動的に挿入されます
		self.fc_units = [self.lstm_units[-1], 768, 1024]
		
		# Batch Normalizationはあまり効果がない？
		self.fc_apply_batchnorm = True
		self.fc_apply_batchnorm_to_input = True
		self.fc_apply_batchnorm_to_output = False
		self.fc_apply_dropout = False
		self.fc_activation_function = "elu"

	def check(self):
		if len(self.lstm_units) < 2:
			raise Exception("You need to add one or more hidden layers to LSTM network.")
		if len(self.fc_units) < 1:
			raise Exception("You need to add one or more hidden layers to fully-connected network.")

config = Config()

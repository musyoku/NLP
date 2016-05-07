# -*- coding: utf-8 -*-
import os, time
import numpy as np
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from bnlstm import BNLSTM

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

class Conf:
	def __init__(self):
		self.use_gpu = True
		self.n_vocab = -1

		# 文字埋め込みベクトルの次元数
		self.embed_size = 200

		# 各LSTMレイヤのユニット数を入力側から出力側に向かって並べる
		# e.g 500(input vector)->250->100(output vector)
		# self.q_fc_units = [500, 250, 100]
		self.lstm_units = [self.embed_size, 1000]
		# if true, it uses BNLSTM
		self.lstm_apply_batchnorm = False
		self.lstm_apply_dropout = False

		# Fully-connected network that converts an output of the LSTM to a label distribution or an embed vector
		# We don't contain output unit size here.
		self.fc_units = [self.lstm_units[-1], 500]
		
		self.fc_apply_batchnorm = False
		self.fc_apply_dropout = False
		self.fc_activation_function = "elu"

		# "embed_vector": outputs an embed vector. Instead of softmax layer, We use EmbedID.reverse() to convert vector to label id.
		# "label_softmax": outputs a probability distribution of label ids using softmax layer
		self.fc_output_type = "embed_vector"

		self.learning_rate = 0.0025
		self.gradient_momentum = 0.95

	def check(self):
		if len(self.lstm_units) < 2:
			raise Exception("You need to add one or more hidden layers to LSTM network.")
		if len(self.fc_units) < 1:
			raise Exception("You need to add one or more hidden layers to fully-connected network.")
		if self.fc_output_type is not in ["embed_vector", "label_softmax"]:
			raise Exception("Invalid output type of the fully-connected layer.")


class LSTMNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(LSTM, self).__init__(**layers)
		self.n_layers = 0
		self.apply_dropout = False

	def forward_one_step(self, x, test):
		chain = [x]
		embed = self.embed_id(chain[-1])
		chain.append(embed)

		# Hidden layers
		for i in range(self.n_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			output = u
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def reset_state(self):
		for i in range(self.n_layers):
			getattr(self, "layer_%i" % i).reset_state()

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class FullyConnectedNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(FullyConnectedNetwork, self).__init__(**layers)
		self.n_layers = 0
		self.activation_function = "tanh"
		self.apply_dropout = False
		self.apply_batchnorm = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			u = chain[-1]
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			u = getattr(self, "layer_%i" % i)(u)
			output = f(u)
			if self.apply_dropout and i != self.n_layers - 1:
				output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class LSTM:
	OUTPUT_TYPE_SOFTMAX = 1
	OUTPUT_TYPE_EMBED_VECTOR = 2
	def __init__(self, conf, name="lstm"):
		conf.check()
		if conf.fc_output_type == "embed_vector":
			self.output_type = self.OUTPUT_TYPE_EMBED_VECTOR
		elif conf.fc_output_type == "label_softmax":
			self.output_type = self.OUTPUT_TYPE_SOFTMAX
		else:
			raise Exception()
		self.embed_id, self.lstm, self.fc = self.build(conf)
		self.name = name
		self.optimizer_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_lstm.setup(self.lstm)
		self.optimizer_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.optimizer_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_fc.setup(self.fc)
		self.optimizer_fc.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.optimizer_embed_id = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_embed_id.setup(self.embed_id)
		self.optimizer_embed_id.add_hook(chainer.optimizer.GradientClipping(10.0))

	def build(conf):
		conf.check()
		wscale = 1.0

		embed_id = L.EmbedID(conf.n_vocab, conf.lstm_units[0])

		lstm_attributes = {}
		lstm_units = zip(conf.lstm_units[:-1], conf.lstm_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		lstm = LSTMNetwork(**lstm_attributes)
		lstm.n_layers = len(lstm_units)
		lstm.apply_dropout = conf.lstm_apply_dropout
		if conf.use_gpu:
			lstm.to_gpu()

		fc_attributes = {}
		fc_units = zip(conf.fc_units[:-1], conf.fc_units[1:])
		if conf.fc_output_type == "embed_vector":
			fc_units += [(conf.fc_units[-1], conf.embed_size)]
		elif conf.fc_output_type == "label_softmax":
			fc_units += [(conf.fc_units[-1], conf.n_vocab)]
		else:
			raise Exception()

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		fc = FullyConnectedNetwork(**fc_attributes)
		fc.n_layers = len(fc_units)
		fc.activation_function = conf.fc_activation_function
		fc.apply_batchnorm = conf.fc_apply_batchnorm
		fc.apply_dropout = conf.fc_apply_dropout
		if conf.use_gpu:
			fc.to_gpu()

		return embed_id, lstm, fc

	def __call__(self, x, test=False, softmax=True):
		output = self.lstm(x, test=test)
		output = self.fc(output, test=test)
		if softmax:
			output = F.softmax(output)
		return output

	@property
	def xp(self):
		return np if self.lstm.layer_0._cpu else cuda.cupy

	def reset_state(self):
		self.lstm.reset_state()

	def predict(self, word, gpu=True, test=True):
		xp = self.xp
		c0 = Variable(xp.asarray([word], dtype=np.int32))
		output = self(c0, test=test, softmax=False)
		ids = xp.argmax(output.data, axis=1)
		return ids

	def distribution(self, word, test=True):
		xp = self.xp
		c0 = Variable(xp.asarray([word], dtype=np.int32))
		output = self(c0, test=test, softmax=True)
		if xp is cuda.cupy:
			output.to_cpu()
		return output.data

	def train(self, seq_batch, gpu=True, test=False):
		self.reset_state()
		xp = self.xp
		sum_loss = 0
		seq_batch = seq_batch.T
		for c0, c1 in zip(seq_batch[:-1], seq_batch[1:]):
			c0[c0 == -1] = 0
			c0 = Variable(xp.asanyarray(c0, dtype=np.int32))
			c1 = Variable(xp.asanyarray(c1, dtype=np.int32))
			output = self(c0, test=test, softmax=False)
			loss = F.softmax_cross_entropy(output, c1)
			sum_loss += loss
		self.optimizer_lstm.zero_grads()
		self.optimizer_fc.zero_grads()
		sum_loss.backward()
		self.optimizer_lstm.update()
		self.optimizer_fc.update()
		return sum_loss.data

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/%s_%s.hdf5" % (self.name, attr)
				if os.path.isfile(filename):
					print "loading",  filename
					serializers.load_hdf5(filename, prop)
				else:
					print filename, "missing."
		print "model loaded."

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				serializers.save_hdf5(dir + "/%s_%s.hdf5" % (self.name, attr), prop)
		print "model saved."

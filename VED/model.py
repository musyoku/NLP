# -*- coding: utf-8 -*-
import os, time
import numpy as np
import chainer
import collections, six
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from bnlstm import BNLSTM
from embed_id import EmbedID

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
		self.char_embed_size = 20
		self.word_embed_size = 200

		self.word_encoder_lstm_units = [500]
		self.word_encoder_lstm_apply_batchnorm = False
		self.word_encoder_fc_hidden_units = []
		self.word_encoder_fc_apply_batchnorm = False
		self.word_encoder_fc_apply_dropout = False
		self.word_encoder_fc_nonlinear = "elu"

		self.word_decoder_lstm_units = [500]
		self.word_decoder_lstm_apply_batchnorm = False
		self.word_decoder_fc_hidden_units = []
		self.word_decoder_fc_apply_batchnorm = False
		self.word_decoder_fc_apply_dropout = False
		self.word_decoder_fc_nonlinear = "elu"
		self.word_decoder_merge_type = "concat"

		self.learning_rate = 0.0025
		self.gradient_momentum = 0.95

	def check(self):
		if len(self.word_encoder_lstm_units) < 1:
			raise Exception("You need to add one or more hidden layers to LSTM network.")

class LSTMNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(LSTMNetwork, self).__init__(**layers)
		self.n_layers = 0

	def forward_one_step(self, x, test):
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			output = getattr(self, "layer_%i" % i)(chain[-1])
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
		self.nonlinear = "elu"
		self.apply_dropout = False
		self.apply_batchnorm = False

	def forward_one_step(self, x, test):
		f = activations[self.nonlinear]
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			u = chain[-1]
			u = getattr(self, "layer_%i" % i)(u)
			if self.apply_batchnorm:
				u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout and i != self.n_layers - 1:
				output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)
	
def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class Model:
	def __init__(self, conf, name="lstm"):
		self.name = name
		self.embed_id, self.word_encoder_lstm, self.word_encoder_fc, self.word_decoder_lstm, self.word_decoder_fc = self.build(conf)

		self.optimizer_word_encoder_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_encoder_lstm.setup(self.forward_lstm)
		self.optimizer_word_encoder_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_word_decoder_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_decoder_lstm.setup(self.backward_lstm)
		self.optimizer_word_decoder_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_word_encoder_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_encoder_fc.setup(self.word_encoder_fc)
		self.optimizer_word_encoder_fc.add_hook(GradientClipping(10.0))

		self.optimizer_word_decoder_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_decoder_fc.setup(self.word_decoder_fc)
		self.optimizer_word_decoder_fc.add_hook(GradientClipping(10.0))

		self.optimizer_embed_id = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_embed_id.setup(self.embed_id)
		self.optimizer_embed_id.add_hook(GradientClipping(10.0))

	def build(self, conf):
		conf.check()
		wscale = 0.1

		embed_id = EmbedID(conf.n_vocab, conf.char_embed_size, ignore_label=-1)
		if conf.use_gpu:
			embed_id.to_gpu()

		# Encoder
		lstm_attributes = {}
		lstm_units = [(conf.char_embed_size, conf.word_encoder_lstm_units[0])]
		lstm_units += zip(conf.word_encoder_lstm_units[:-1], conf.word_encoder_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.word_encoder_lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		word_encoder_lstm = LSTMNetwork(**lstm_attributes)
		word_encoder_lstm.n_layers = len(lstm_units)
		if conf.use_gpu:
			word_encoder_lstm.to_gpu()

		# Decoder
		lstm_attributes = {}
		lstm_units = [(conf.word_embed_size + conf.char_embed_size, conf.word_encoder_lstm_units[0])]
		lstm_units += zip(conf.word_encoder_lstm_units[:-1], conf.word_encoder_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.word_encoder_lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		word_decoder_lstm = LSTMNetwork(**lstm_attributes)
		word_decoder_lstm.n_layers = len(lstm_units)
		if conf.use_gpu:
			word_decoder_lstm.to_gpu()

		# Variational encoder
		fc_attributes = {}
		fc_units = []
		if len(conf.word_encoder_fc_hidden_units) > 0:
			fc_units = [(conf.word_encoder_lstm_units[-1], conf.word_encoder_fc_hidden_units[0])]
			fc_units += zip(conf.word_encoder_fc_hidden_units[:-1], conf.word_encoder_fc_hidden_units[1:])
			for i, (n_in, n_out) in enumerate(fc_units):
				fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
				fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			fc_attributes["layer_mean"] = L.Linear(conf.word_encoder_fc_hidden_units[-1], conf.word_embed_size, wscale=wscale)
			fc_attributes["layer_var"] = L.Linear(conf.word_encoder_fc_hidden_units[-1], conf.word_embed_size, wscale=wscale)
		else:
			fc_attributes["layer_mean"] = L.Linear(conf.word_encoder_lstm_units[-1], conf.word_embed_size, wscale=wscale)
			fc_attributes["layer_var"] = L.Linear(conf.word_encoder_lstm_units[-1], conf.word_embed_size, wscale=wscale)

		word_encoder_fc = FullyConnectedNetwork(**fc_attributes)
		word_encoder_fc.n_layers = len(fc_units)
		word_encoder_fc.nonlinear = conf.word_encoder_fc_nonlinear
		word_encoder_fc.apply_batchnorm = conf.word_encoder_fc_apply_batchnorm
		word_encoder_fc.apply_dropout = conf.word_encoder_fc_apply_dropout
		if conf.use_gpu:
			word_encoder_fc.to_gpu()

		# ID decoder
		fc_attributes = {}
		fc_units = []
		if len(conf.word_decoder_fc_hidden_units) > 0:
			fc_units = [(conf.word_decoder_lstm_units[-1], conf.word_decoder_fc_hidden_units[0])]
			fc_units += zip(conf.word_decoder_fc_hidden_units[:-1], conf.word_decoder_fc_hidden_units[1:])
			fc_units += zip(conf.word_decoder_fc_hidden_units[-1], conf.char_embed_size)
		else:
			fc_units = [(conf.word_decoder_lstm_units[-1], conf.char_embed_size)]

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

		word_decoder_fc = FullyConnectedNetwork(**fc_attributes)
		word_decoder_fc.n_layers = len(fc_units)
		word_decoder_fc.nonlinear = conf.word_encoder_fc_nonlinear
		word_decoder_fc.apply_batchnorm = conf.word_decoder_fc_apply_batchnorm
		word_decoder_fc.apply_dropout = conf.word_decoder_fc_apply_dropout
		if conf.use_gpu:
			word_decoder_fc.to_gpu()

		return embed_id, word_encoder_lstm, word_encoder_fc, word_decoder_lstm, word_decoder_fc

	@property
	def xp(self):
		return np if self.forward_lstm.layer_0._cpu else cuda.cupy

	@property
	def gpu(self):
		return True if self.xp is cuda.cupy else False

	def reset_state(self):
		self.forward_lstm.reset_state()
		self.backward_lstm.reset_state()
		self.prev_forget = None

	def zero_grads(self):
		self.optimizer_word_encoder_lstm.zero_grads()
		self.optimizer_word_encoder_fc.zero_grads()
		self.optimizer_word_decoder_lstm.zero_grads()
		self.optimizer_word_decoder_fc.zero_grads()
		self.optimizer_embed_id.zero_grads()

	def update(self):
		self.optimizer_word_encoder_lstm.update()
		self.optimizer_word_encoder_fc.update()
		self.optimizer_word_decoder_lstm.update()
		self.optimizer_word_decoder_fc.update()
		self.optimizer_embed_id.update()

	def should_save(self, prop):
		if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod) or isinstance(prop, EmbedID):
			return True
		return False

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if self.should_save(prop):
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
			if self.should_save(prop):
				serializers.save_hdf5(dir + "/%s_%s.hdf5" % (self.name, attr), prop)
		print "model saved."

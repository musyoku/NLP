# -*- coding: utf-8 -*-
import os, time, math
import numpy as np
import chainer
import collections, six
import sgu
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
		self.gpu_enabled = True
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
		self.word_decoder_merge_type = "concat"

		self.word_ngram_lstm_units = [500]
		self.word_ngram_fc_hidden_units = []
		self.word_ngram_fc_apply_batchnorm = False
		self.word_ngram_fc_apply_dropout = False
		self.word_ngram_fc_nonlinear = "elu"

		self.discriminator_hidden_units = [500]
		self.discriminator_apply_batchnorm = False
		self.discriminator_apply_dropout = False
		self.discriminator_nonlinear = "elu"

		self.learning_rate = 0.0003
		self.gradient_momentum = 0.95

	def check(self):
		if len(self.word_encoder_lstm_units) < 1:
			raise Exception("You need to add one or more hidden layers to LSTM network.")

class LSTM(L.LSTM):

	def __call__(self, x, condition=None):
		lstm_in = self.upward(x)
		if self.h is not None:
			lstm_in += self.lateral(self.h)
		if self.c is None:
			xp = self.xp
			self.c = Variable(xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),	volatile="auto")
		if condition is None:
			self.c, self.h = F.lstm(self.c, lstm_in)
		else:
			c, h = F.lstm(self.c, lstm_in)
			if self.h is None:
				self.h = h
				self.c = c
			else:
				self.h = F.where(condition, h, self.h)
				self.c = F.where(condition, c, self.c)
		return self.h

class GRU(L.StatefulGRU):

	def __call__(self, x, condition=None):
		z = self.W_z(x)
		h_bar = self.W(x)
		if self.h is not None:
			r = F.sigmoid(self.W_r(x) + self.U_r(self.h))
			z += self.U_z(self.h)
			h_bar += self.U(r * self.h)
		z = F.sigmoid(z)
		h_bar = F.tanh(h_bar)

		h_new = z * h_bar
		if self.h is not None:
			h_new += (1 - z) * self.h
		if condition is None:
			self.h = h_new
		else:
			if self.h is None:
				self.h = h_new
			else:
				self.h = F.where(condition, h_new, self.h)
		return self.h

class DSGU(sgu.StatefulDSGU):

	def __call__(self, x, condition=None):

		if self.h is None:
			z_t = sgu.hard_sigmoid(self.W_xz(x))
			h_t = z_t * 0.5
		else:
			h_t = sgu.DSGU.__call__(self, self.h, x)

		if condition is None:
			self.h = h_t
		else:
			if self.h is None:
				self.h = h_t
			else:
				self.h = F.where(condition, h_t, self.h)
		return h_t

class LSTMEncoder(chainer.Chain):
	def __init__(self, **layers):
		super(LSTMEncoder, self).__init__(**layers)
		self.n_layers = 0

	def forward_one_step(self, x, condition, test):
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			output = getattr(self, "layer_%i" % i)(chain[-1], condition)
			chain.append(output)

		return chain[-1]

	def reset_state(self):
		for i in range(self.n_layers):
			getattr(self, "layer_%i" % i).reset_state()

	def __call__(self, x, condition=None, test=False):
		return self.forward_one_step(x, condition, test=test)

class LSTMDecoder(LSTMEncoder):

	def forward_one_step(self, x, test):
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			output = getattr(self, "layer_%i" % i)(chain[-1])
			chain.append(output)

		output = getattr(self, "layer_output")(chain[-1])
		chain.append(output)

		# bias = cuda.cupy.zeros(output.data.shape, dtype=np.float32)
		# bias[:, 0] = 1.0
		# bias = Variable(bias)
		# output = chain[-1] + bias
		# chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class MultiLayerPerceptron(chainer.Chain):
	def __init__(self, **layers):
		super(MultiLayerPerceptron, self).__init__(**layers)
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

class GaussianNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(GaussianNetwork, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = False
		self.apply_batchnorm = False
		self.apply_dropout = False
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test=False, apply_f=True):
		f = activations[self.activation_function]

		chain = [x]

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		u = chain[-1]
		mean = self.layer_mean(u)

		# log(sigma^2)
		u = chain[-1]
		ln_var = self.layer_var(u)

		return mean, ln_var

	def __call__(self, x, test=False, apply_f=True):
		mean, ln_var = self.forward_one_step(x, test=test, apply_f=apply_f)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

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
		self.char_embed_size = conf.char_embed_size
		self.word_embed_size = conf.word_embed_size
		self.conf = conf

		self.embed_id, self.word_encoder_lstm, self.word_encoder_fc, self.word_decoder_lstm, self.discriminator, self.word_ngram_lstm, self.word_ngram_fc = self.build(conf)

		self.optimizer_word_encoder_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_encoder_lstm.setup(self.word_encoder_lstm)
		self.optimizer_word_encoder_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_word_decoder_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_decoder_lstm.setup(self.word_decoder_lstm)
		self.optimizer_word_decoder_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_word_encoder_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_encoder_fc.setup(self.word_encoder_fc)
		self.optimizer_word_encoder_fc.add_hook(GradientClipping(10.0))

		self.optimizer_discriminator = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator.setup(self.discriminator)
		self.optimizer_discriminator.add_hook(GradientClipping(10.0))

		self.optimizer_word_ngram_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_ngram_lstm.setup(self.word_ngram_lstm)
		self.optimizer_word_ngram_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_word_ngram_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_word_ngram_fc.setup(self.word_ngram_fc)
		self.optimizer_word_ngram_fc.add_hook(GradientClipping(10.0))

		self.optimizer_embed_id = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_embed_id.setup(self.embed_id)
		self.optimizer_embed_id.add_hook(GradientClipping(10.0))

	def build(self, conf):
		conf.check()
		wscale = 0.1

		embed_id = EmbedID(conf.n_vocab, conf.char_embed_size, ignore_label=-1)
		if conf.gpu_enabled:
			embed_id.to_gpu()

		# encoder
		lstm_attributes = {}
		lstm_units = [(conf.char_embed_size, conf.word_encoder_lstm_units[0])]
		lstm_units += zip(conf.word_encoder_lstm_units[:-1], conf.word_encoder_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.word_encoder_lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = LSTM(n_in, n_out)

		word_encoder_lstm = LSTMEncoder(**lstm_attributes)
		word_encoder_lstm.n_layers = len(lstm_units)
		if conf.gpu_enabled:
			word_encoder_lstm.to_gpu()

		# decoder
		lstm_attributes = {}
		lstm_units = [(conf.char_embed_size + conf.word_embed_size, conf.word_decoder_lstm_units[0])]
		lstm_units += zip(conf.word_decoder_lstm_units[:-1], conf.word_decoder_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.word_encoder_lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = LSTM(n_in, n_out)
		lstm_attributes["layer_output"] = L.Linear(conf.word_decoder_lstm_units[-1], conf.n_vocab, wscale=wscale)

		word_decoder_lstm = LSTMDecoder(**lstm_attributes)
		word_decoder_lstm.n_layers = len(lstm_units)
		if conf.gpu_enabled:
			word_decoder_lstm.to_gpu()

		# word n-gram
		lstm_attributes = {}
		lstm_units = [(conf.word_embed_size, conf.word_ngram_lstm_units[0])]
		lstm_units += zip(conf.word_ngram_lstm_units[:-1], conf.word_ngram_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.word_encoder_lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = LSTM(n_in, n_out)

		word_ngram_lstm = LSTMEncoder(**lstm_attributes)
		word_ngram_lstm.n_layers = len(lstm_units)
		if conf.gpu_enabled:
			word_ngram_lstm.to_gpu()

		# variational encoder for word n-gram
		fc_attributes = {}
		fc_units = []
		if len(conf.word_ngram_fc_hidden_units) > 0:
			fc_units = [(conf.word_ngram_lstm_units[-1], conf.word_ngram_fc_hidden_units[0])]
			fc_units += zip(conf.word_ngram_fc_hidden_units[:-1], conf.word_ngram_fc_hidden_units[1:])
			for i, (n_in, n_out) in enumerate(fc_units):
				fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
				fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			fc_attributes["layer_mean"] = L.Linear(conf.word_ngram_fc_hidden_units[-1], conf.word_embed_size, wscale=wscale)
			fc_attributes["layer_var"] = L.Linear(conf.word_ngram_fc_hidden_units[-1], conf.word_embed_size, wscale=wscale)
		else:
			fc_attributes["layer_mean"] = L.Linear(conf.word_ngram_lstm_units[-1], conf.word_embed_size, wscale=wscale)
			fc_attributes["layer_var"] = L.Linear(conf.word_ngram_lstm_units[-1], conf.word_embed_size, wscale=wscale)

		word_ngram_fc = GaussianNetwork(**fc_attributes)
		word_ngram_fc.n_layers = len(fc_units)
		word_ngram_fc.nonlinear = conf.word_ngram_fc_nonlinear
		word_ngram_fc.apply_batchnorm = conf.word_ngram_fc_apply_batchnorm
		word_ngram_fc.apply_dropout = conf.word_ngram_fc_apply_dropout
		if conf.gpu_enabled:
			word_ngram_fc.to_gpu()

		# variational encoder
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

		word_encoder_fc = GaussianNetwork(**fc_attributes)
		word_encoder_fc.n_layers = len(fc_units)
		word_encoder_fc.nonlinear = conf.word_encoder_fc_nonlinear
		word_encoder_fc.apply_batchnorm = conf.word_encoder_fc_apply_batchnorm
		word_encoder_fc.apply_dropout = conf.word_encoder_fc_apply_dropout
		if conf.gpu_enabled:
			word_encoder_fc.to_gpu()

		# discriminator
		fc_attributes = {}
		fc_units = [(conf.word_embed_size, conf.discriminator_hidden_units[0])]
		fc_units += zip(conf.discriminator_hidden_units[:-1], conf.discriminator_hidden_units[1:])
		fc_units += [(conf.discriminator_hidden_units[-1], 2)]
		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

		discriminator = MultiLayerPerceptron(**fc_attributes)
		discriminator.n_layers = len(fc_units)
		discriminator.nonlinear = conf.word_encoder_fc_nonlinear
		discriminator.apply_batchnorm = conf.word_encoder_fc_apply_batchnorm
		discriminator.apply_dropout = conf.word_encoder_fc_apply_dropout
		if conf.gpu_enabled:
			discriminator.to_gpu()

		return embed_id, word_encoder_lstm, word_encoder_fc, word_decoder_lstm, discriminator, word_ngram_lstm, word_ngram_fc

	@property
	def xp(self):
		return np if self.word_encoder_lstm.layer_0._cpu else cuda.cupy

	@property
	def gpu_enabled(self):
		return True if self.xp is cuda.cupy else False

	def encode_word(self, char_ids, test=False):
		xp = self.xp
		self.word_encoder_lstm.reset_state()
		output = None
		for i in xrange(len(char_ids)):
			c0 = Variable(xp.asanyarray([char_ids[i]], dtype=xp.int32))
			c0 = self.embed_id(c0)
			output = self.word_encoder_lstm(c0, test=test)
		output = self.word_encoder_fc(output, apply_f=True)
		return output

	def encode_word_batch(self, char_ids_batch, test=False):
		xp = self.xp
		self.word_encoder_lstm.reset_state()
		output = None
		char_ids_batch = char_ids_batch.T
		batchsize = char_ids_batch.shape[1]
		for i in xrange(char_ids_batch.shape[0]):
			condition_args = np.argwhere(char_ids_batch[i] == -1).reshape(-1)
			condition = np.full((batchsize, self.conf.word_encoder_lstm_units[0]), True, dtype=np.bool)
			for j in xrange(condition_args.shape[0]):
				condition[condition_args[j], :] = False
			condition = Variable(condition)
			if self.gpu_enabled:
				condition.to_gpu()
			c0 = Variable(xp.asanyarray(char_ids_batch[i], dtype=xp.int32))
			c0 = self.embed_id(c0)
			output = self.word_encoder_lstm(c0, condition, test=test)
		output = self.word_encoder_fc(output, apply_f=True)
		return output

	def decode_word(self, word_vec, test=False, argmax=True):
		if word_vec.ndim != 1:
			raise Exception()
		xp = self.xp
		self.word_decoder_lstm.reset_state()
		word_vec = Variable(xp.asanyarray([word_vec], dtype=xp.float32))
		prev_y = None
		char_ids = []
		for i in xrange(100):
			if prev_y is None:
				prev_y = Variable(xp.zeros((1, self.char_embed_size), dtype=xp.float32))
			dec_in = F.concat((word_vec, prev_y))
			y = self.word_decoder_lstm(dec_in, test=test)
			if argmax:
				ids = xp.argmax(y.data, axis=1)
			else:
				y = F.softmax(y)
				y.to_cpu()
				ids = [np.random.choice(np.arange(y.data.shape[1]), p=y.data[0])]
			char_ids.append(int(ids[0]))
			if ids[0] == 0:
				return char_ids
			ids = Variable(xp.asanyarray(ids, dtype=xp.int32))
			prev_y = self.embed_id(ids)
		return char_ids

	def decode_word_batch(self, word_vec_batch, test=False):
		xp = self.xp
		self.word_decoder_lstm.reset_state()
		if isinstance(word_vec_batch, Variable) == False:
			word_vec_batch = Variable(word_vec_batch)
		batchsize = word_vec_batch.data.shape[0]
		prev_y = None
		length_limit = 100
		char_ids = xp.full((batchsize, length_limit), -1, dtype=xp.int32)
		for i in xrange(length_limit):
			if prev_y is None:
				prev_y = Variable(xp.zeros((batchsize, self.char_embed_size), dtype=xp.float32))
			dec_in = F.concat((word_vec_batch, prev_y))
			y = self.word_decoder_lstm(dec_in, test=test)
			ids = xp.argmax(y.data, axis=1)
			char_ids[:,i] = ids
			ids = Variable(ids)
			prev_y = self.embed_id(ids)
		if self.gpu_enabled:
			char_ids = cuda.to_cpu(char_ids)
		return char_ids

	def sample_z(self, batchsize, z_dim):
		z = np.random.normal(0, 1, (batchsize, z_dim)).astype(np.float32)
		z = Variable(z)
		if self.gpu_enabled:
			z.to_gpu()
		return z

	def train_word_embedding(self, char_ids):
		xp = self.xp
		word_vec = self.encode_word(char_ids)

		# reconstruction loss
		loss_reconstruction = 0
		self.word_decoder_lstm.reset_state()
		prev_y = None
		for i in xrange(len(char_ids)):
			if prev_y is None:
				prev_y = Variable(xp.zeros((1, self.char_embed_size), dtype=xp.float32))
			dec_in = F.concat((word_vec, prev_y))
			y = self.word_decoder_lstm(dec_in, test=False)
			target = Variable(xp.asanyarray([char_ids[i]], dtype=xp.int32))
			loss = F.softmax_cross_entropy(y, target)
			prev_y = self.embed_id(target)
			loss_reconstruction += loss

		self.zero_grads_generator()
		loss_reconstruction.backward()
		self.update_generator()


		# adversarial loss
		## 0: from encoder
		## 1: from noise
		real_z = self.sample_z(1, self.word_embed_size)
		fake_z = word_vec
		y_fake = self.discriminator(fake_z, test=False)

		## train generator
		loss_generator = F.softmax_cross_entropy(y_fake, Variable(xp.ones(1, dtype=xp.int32)))

		self.zero_grads_generator()
		loss_generator.backward()
		self.update_generator()

		# train discriminator
		y_real = self.discriminator(real_z, test=False)
		loss_discriminator = F.softmax_cross_entropy(y_fake, Variable(xp.zeros(1, dtype=xp.int32)))
		loss_discriminator += F.softmax_cross_entropy(y_real, Variable(xp.ones(1, dtype=xp.int32)))

		self.optimizer_discriminator.zero_grads()
		loss_discriminator.backward()
		self.optimizer_discriminator.update()

		return float(loss_reconstruction.data), float(loss_generator.data), float(loss_discriminator.data)

	def train_word_embedding_batch(self, char_ids_batch):
		xp = self.xp
		word_vec = self.encode_word_batch(char_ids_batch)
		batchsize = char_ids_batch.shape[0]
		char_ids_batch = char_ids_batch.T

		# reconstruction loss
		loss_reconstruction = 0
		self.word_decoder_lstm.reset_state()
		prev_y = None
		for i in xrange(char_ids_batch.shape[0]):
			if prev_y is None:
				prev_y = Variable(xp.zeros((batchsize, self.char_embed_size), dtype=xp.float32))
			dec_in = F.concat((word_vec, prev_y))
			y = self.word_decoder_lstm(dec_in, test=False)
			target = Variable(char_ids_batch[i])
			if self.gpu_enabled:
				target.to_gpu()
			loss = F.softmax_cross_entropy(y, target)
			prev_y = self.embed_id(target)
			loss_reconstruction += loss

		self.zero_grads_generator()
		loss_reconstruction.backward()
		self.update_generator()

		# adversarial loss
		## 0: from encoder
		## 1: from noise
		real_z = self.sample_z(batchsize, self.word_embed_size)
		fake_z = word_vec
		y_fake = self.discriminator(fake_z, test=False)

		## train generator
		loss_generator = F.softmax_cross_entropy(y_fake, Variable(xp.ones((batchsize,), dtype=xp.int32)))

		self.zero_grads_generator()
		loss_generator.backward()
		self.update_generator()

		# train discriminator
		y_real = self.discriminator(real_z, test=False)
		loss_discriminator = F.softmax_cross_entropy(y_fake, Variable(xp.zeros((batchsize,), dtype=xp.int32)))
		loss_discriminator += F.softmax_cross_entropy(y_real, Variable(xp.ones((batchsize,), dtype=xp.int32)))

		self.optimizer_discriminator.zero_grads()
		loss_discriminator.backward()
		self.optimizer_discriminator.update()

		return float(loss_reconstruction.data), float(loss_generator.data), float(loss_discriminator.data)

	def gaussian_nll_keepbatch(self, x, mean, ln_var, clip=True):
		if clip:
			clip_min = math.log(0.001)
			clip_max = math.log(10)
			ln_var = F.clip(ln_var, clip_min, clip_max)
		x_prec = F.exp(-ln_var)
		x_diff = x - mean
		x_power = (x_diff * x_diff) * x_prec * 0.5
		return F.sum((math.log(2.0 * math.pi) + ln_var) * 0.5 + x_power, axis=1)

	def gaussian_kl_divergence_keepbatch(self, mean, ln_var):
		var = F.exp(ln_var)
		kld = F.sum(mean * mean + var - ln_var - 1, axis=1) * 0.5
		return kld

	def train_word_ngram_batch(self, word_vec_batch, next_word_vec_batch):
		output = self.word_ngram_lstm(word_vec_batch, test=False)
		mean, ln_var = self.word_ngram_fc(output, apply_f=False)
		nll = self.gaussian_nll_keepbatch(next_word_vec_batch, mean, ln_var)
		kld = self.gaussian_kl_divergence_keepbatch(mean, ln_var)
		loss = F.sum(loss + kld)

		self.zero_grads_word_ngram()
		loss.backward()
		self.update_word_ngram()

	def Pw_h(self, word_char_ids, context_char_ids):
		word_vec = self.encode_word(word_char_ids, test=True)
		context_vec = self.encode_word(context_char_ids, test=True)
		self.word_ngram_lstm.reset_state()

		output = self.word_ngram_lstm(context_vec, test=True)
		mean, ln_var = self.word_ngram_fc(output, apply_f=False)
		log_likelihood = -self.gaussian_nll_keepbatch(word_vec, mean, ln_var)
		return math.exp(float(log_likelihood.data))

	def Pw_h_batch(self, word_char_ids_batch, context_char_ids_batch):
		word_vec = self.encode_word_batch(word_char_ids_batch, test=True)
		context_vec = self.encode_word_batch(context_char_ids_batch, test=True)
		self.word_ngram_lstm.reset_state()

		output = self.word_ngram_lstm(context_vec, test=True)
		mean, ln_var = self.word_ngram_fc(output, apply_f=False)
		log_likelihood = -self.gaussian_nll_keepbatch(word_vec, mean, ln_var)
		likelihood = F.exp(log_likelihood)
		if self.gpu_enabled:
			likelihood.to_cpu()
		return likelihood.data

	def reset_state(self):
		self.word_encoder_lstm.reset_state()
		self.word_decoder_lstm.reset_state()

	def zero_grads_word_ngram():
		self.optimizer_word_ngram_lstm.zero_grads()
		self.optimizer_word_ngram_fc.zero_grads()

	def update_word_ngram():
		self.optimizer_word_ngram_lstm.update()
		self.optimizer_word_ngram_fc.update()

	def zero_grads_generator(self):
		self.optimizer_word_encoder_lstm.zero_grads()
		self.optimizer_word_encoder_fc.zero_grads()
		self.optimizer_word_decoder_lstm.zero_grads()
		self.optimizer_embed_id.zero_grads()

	def update_generator(self):
		self.optimizer_word_encoder_lstm.update()
		self.optimizer_word_encoder_fc.update()
		self.optimizer_word_decoder_lstm.update()
		self.optimizer_embed_id.update()

	def zero_grads(self):
		self.zero_grads_generator()
		self.optimizer_discriminator.zero_grads()

	def update(self):
		self.update_generator()
		self.optimizer_discriminator.update()

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

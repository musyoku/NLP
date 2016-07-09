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
		self.embed_size = 200

		self.lstm_hidden_units = [500]
		# if true, it uses BNLSTM
		self.lstm_apply_batchnorm = False
		self.lstm_apply_dropout = False

		self.fc_hidden_units = [500]
		self.fc_apply_batchnorm = False
		self.fc_apply_dropout = True
		self.fc_activation_function = "elu"
		self.fc_output_type = LSTM.OUTPUT_TYPE_SOFTMAX

		self.forget_hidden_units = [500]
		self.forget_apply_batchnorm = False
		self.forget_apply_dropout = True
		self.forget_activation_function = "elu"

		self.learning_rate = 0.0025
		self.gradient_momentum = 0.95

	def check(self):
		if len(self.lstm_hidden_units) < 1:
			raise Exception("You need to add one or more hidden layers to LSTM network.")

class LSTMNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(LSTMNetwork, self).__init__(**layers)
		self.n_layers = 0
		self.apply_dropout = False

	def forward_one_step(self, x, test):
		chain = [x]

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

class FLSTM(L.LSTM):

	def __call__(self, x):
		lstm_in = self.upward(x)
		if self.h is not None:
			lstm_in += self.lateral(self.h)
		if self.c is None:
			xp = self.xp
			self.c = Variable(xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),	volatile="auto")
		self.c, self.h = F.lstm(self.c, lstm_in)
		return self.h

class Attention(function.Function):
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		context_type, weight_type = in_types

		type_check.expect(
			context_type.dtype == np.float32,
			weight_type.dtype == np.float32,
			context_type.ndim == 2,
			weight_type.ndim == 2,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		context, weight = inputs
		output = context * weight[:, self.index].reshape(-1, 1)
		return output,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(inputs[0])
		context, weight = inputs
		weight = weight[:, self.index].reshape(-1, 1)
		z = xp.zeros((context.shape[0], 2), dtype=xp.float32)
		z[:, self.index] = xp.sum(grad_outputs[0] * context, axis=1)
		return grad_outputs[0] * weight, z

def apply_attention(context, weight, index=0):
	attention = Attention()
	attention.index = index
	return attention(context, weight)
	
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

class LSTM:
	OUTPUT_TYPE_SOFTMAX = 1
	OUTPUT_TYPE_EMBED_VECTOR = 2
	def __init__(self, conf, name="lstm"):
		self.output_type = conf.fc_output_type
		self.embed_id, self.forward_lstm, self.backward_lstm, self.fc, self.forget = self.build(conf)
		self.name = name

		self.optimizer_forward_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_forward_lstm.setup(self.forward_lstm)
		self.optimizer_forward_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_backward_lstm = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_backward_lstm.setup(self.backward_lstm)
		self.optimizer_backward_lstm.add_hook(GradientClipping(10.0))

		if self.fc is not None:
			self.optimizer_fc = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
			self.optimizer_fc.setup(self.fc)
			self.optimizer_fc.add_hook(GradientClipping(10.0))

		self.optimizer_forget = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_forget.setup(self.forget)
		self.optimizer_forget.add_hook(GradientClipping(10.0))

		self.optimizer_embed_id = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_embed_id.setup(self.embed_id)
		self.optimizer_embed_id.add_hook(GradientClipping(10.0))

	def build(self, conf):
		conf.check()
		wscale = 1.0

		embed_id = EmbedID(conf.n_vocab, conf.embed_size, ignore_label=-1)
		if conf.use_gpu:
			embed_id.to_gpu()

		lstm_attributes = {}
		lstm_units = [(conf.embed_size, conf.lstm_hidden_units[0])]
		lstm_units += zip(conf.lstm_hidden_units[:-1], conf.lstm_hidden_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = FLSTM(n_in, n_out)

		forward_lstm = LSTMNetwork(**lstm_attributes)
		forward_lstm.n_layers = len(lstm_units)
		forward_lstm.apply_dropout = conf.lstm_apply_dropout
		if conf.use_gpu:
			forward_lstm.to_gpu()

		lstm_attributes = {}
		lstm_units = [(conf.embed_size, conf.lstm_hidden_units[0])]
		lstm_units += zip(conf.lstm_hidden_units[:-1], conf.lstm_hidden_units[1:])

		for i, (n_in, n_out) in enumerate(lstm_units):
			if conf.lstm_apply_batchnorm:
				lstm_attributes["layer_%i" % i] = BNLSTM(n_in, n_out)
			else:
				lstm_attributes["layer_%i" % i] = FLSTM(n_in, n_out)
		backward_lstm = LSTMNetwork(**lstm_attributes)
		backward_lstm.n_layers = len(lstm_units)
		backward_lstm.apply_dropout = conf.lstm_apply_dropout
		if conf.use_gpu:
			backward_lstm.to_gpu()

		if len(conf.fc_hidden_units) > 0:
			fc_attributes = {}
			fc_units = [(conf.lstm_hidden_units[-1], conf.fc_hidden_units[0])]
			fc_units += zip(conf.fc_hidden_units[:-1], conf.fc_hidden_units[1:])
			if conf.fc_output_type == self.OUTPUT_TYPE_EMBED_VECTOR:
				fc_units += [(conf.fc_hidden_units[-1], conf.embed_size)]
			elif conf.fc_output_type == self.OUTPUT_TYPE_SOFTMAX:
				fc_units += [(conf.fc_hidden_units[-1], conf.n_vocab)]
			else:
				raise Exception()

			for i, (n_in, n_out) in enumerate(fc_units):
				fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
				fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

			fc = FullyConnectedNetwork(**fc_attributes)
			fc.n_layers = len(fc_units)
			fc.activation_function = conf.fc_activation_function
			fc.apply_batchnorm = conf.fc_apply_batchnorm
			fc.apply_dropout = conf.fc_apply_dropout
			if conf.use_gpu:
				fc.to_gpu()
		else:
			fc = None

		forget_attributes = {}
		forget_units = [(conf.lstm_hidden_units[-1] * 2, conf.forget_hidden_units[0])]
		forget_units += zip(conf.forget_hidden_units[:-1], conf.forget_hidden_units[1:])
		forget_units += [(conf.forget_hidden_units[-1], 2)]

		for i, (n_in, n_out) in enumerate(forget_units):
			forget_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			forget_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

		forget = FullyConnectedNetwork(**forget_attributes)
		forget.n_layers = len(forget_units)
		forget.activation_function = conf.forget_activation_function
		forget.apply_batchnorm = conf.forget_apply_batchnorm
		forget.apply_dropout = conf.forget_apply_dropout
		if conf.use_gpu:
			forget.to_gpu()

		return embed_id, forward_lstm, backward_lstm, fc, forget

	def __call__(self, x, test=False, softmax=True):
		output = self.embed_id(x)
		output = self.lstm(output, self.prev_forget, test=test)
		self.prev_forget = F.sigmoid(self.forget(output))
		if self.fc is not None:
			output = self.fc(output, test=test)
		if softmax and self.output_type == self.OUTPUT_TYPE_SOFTMAX:
			output = F.softmax(output)
		return output

	def forward_lstm_one_step(self, x, test=False):
		output = self.embed_id(x)
		output = self.forward_lstm(output, test=test)
		return output

	def backward_lstm_one_step(self, x, test=False):
		output = self.embed_id(x)
		output = self.backward_lstm(output, test=test)
		return output

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

	def predict(self, word, test=True, argmax=False):
		xp = self.xp
		c0 = Variable(xp.asarray([word], dtype=np.int32))
		if self.output_type == self.OUTPUT_TYPE_SOFTMAX:
			output = self(c0, test=test, softmax=True)
			if xp is cuda.cupy:
				output.to_cpu()
			if argmax:
				ids = np.argmax(output.data, axis=1)
			else:
				ids = [np.random.choice(np.arange(output.data.shape[1]), p=output.data[0])]
		elif self.output_type == self.OUTPUT_TYPE_EMBED_VECTOR:
			output = self(c0, test=test, softmax=False)
			if argmax:
				ids = self.embed_id.reverse(output.data, to_cpu=True, sample=False)
			else:
				ids = self.embed_id.reverse(output.data, to_cpu=True, sample=True)
		return ids[0]

	def predict_all(self, seq_batch, test=True, argmax=True):
		self.reset_state()
		forward_h, backward_h = self.scan(seq_batch, test=test)
		xp = self.xp
		seq_batch = seq_batch.T
		result = []
		forgets = []

		for i in xrange(len(forward_h)):
			fh = forward_h[i]
			bh = backward_h[i]
			c = seq_batch[i]
			c = Variable(xp.asanyarray(c, dtype=np.int32))
			if fh is None:
				fh = Variable(xp.zeros(bh.data.shape, dtype=xp.float32))
			if bh is None:
				bh = Variable(xp.zeros(fh.data.shape, dtype=xp.float32))
			h = F.concat((fh, bh))
			forget = self.forget(h, test=test)
			forget = F.softmax(forget)
			out = apply_attention(fh, forget, 0) + apply_attention(bh, forget, 1)
			if self.fc is not None:
				out = self.fc(out, test=test)

			out = F.softmax(out)
			if xp is cuda.cupy:
				out.to_cpu()
			if argmax:
				ids = np.argmax(out.data, axis=1)
			else:
				ids = [np.random.choice(np.arange(out.data.shape[1]), p=output.data[0])]
			result.append(ids)
			forgets.append(forget)
		return result, forgets

	def distribution(self, word, test=True):
		xp = self.xp
		c0 = Variable(xp.asarray([word], dtype=np.int32))
		output = self(c0, test=test, softmax=True)
		if xp is cuda.cupy:
			output.to_cpu()
		return output.data

	def train(self, seq_batch, test=False):
		self.reset_state()
		forward_h, backward_h = self.scan(seq_batch, test=test)
		xp = self.xp
		sum_loss = 0
		seq_batch = seq_batch.T

		for i in xrange(len(forward_h)):
			fh = forward_h[i]
			bh = backward_h[i]
			c = seq_batch[i]
			c = Variable(xp.asanyarray(c, dtype=np.int32))
			if fh is None:
				out = bh
			elif bh is None:
				out = fh
			else:
				h = F.concat((fh, bh))
				forget = self.forget(h, test=test)
				forget = F.softmax(forget)
				out = apply_attention(fh, forget, 0) + apply_attention(bh, forget, 1)
			if self.fc is not None:
				out = self.fc(out, test=test)
			entropy = 0
			if fh is not None and bh is not None:
				entropy = -forget * F.log(forget + 1e-6)
				entropy = F.sum(entropy)
			loss = F.softmax_cross_entropy(out, c) + entropy
			sum_loss += loss

		self.zero_grads()
		sum_loss.backward()
		self.update()
		if self.gpu:
			sum_loss.to_cpu()
		return sum_loss.data

	def scan(self, seq_batch, test=False):
		forward_h = []
		backward_h = []
		self.reset_state()
		xp = self.xp
		seq_batch = seq_batch.T

		forward_h.append(None)
		for i in xrange(seq_batch.shape[0] - 1):
			c0 = seq_batch[i]
			c0 = Variable(xp.asanyarray(c0, dtype=np.int32))
			output = self.forward_lstm_one_step(c0, test=test)
			forward_h.append(output)

		backward_h.append(None)
		for i in xrange(seq_batch.shape[0] - 1):
			c0 = seq_batch[-i-1]
			c0 = Variable(xp.asanyarray(c0, dtype=np.int32))
			output = self.backward_lstm_one_step(c0, test=test)
			backward_h.append(output)

		backward_h.reverse()
		return forward_h, backward_h

	def zero_grads(self):
		self.optimizer_forward_lstm.zero_grads()
		self.optimizer_backward_lstm.zero_grads()
		if self.fc is not None:
			self.optimizer_fc.zero_grads()
		self.optimizer_embed_id.zero_grads()
		self.optimizer_forget.zero_grads()

	def update(self):
		self.optimizer_forward_lstm.update()
		self.optimizer_backward_lstm.update()
		if self.fc is not None:
			self.optimizer_fc.update()
		self.optimizer_embed_id.update()
		self.optimizer_forget.update()

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

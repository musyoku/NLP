# -*- coding: utf-8 -*-
import os, time, re, collections, six
import numpy as np
import chainer
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from config import config
from activations import activations

class StackedLSTM(chainer.Chain):
	def __init__(self, **layers):
		super(StackedLSTM, self).__init__(**layers)
		self.n_layers = 0
		self.activation_function = None
		self.apply_dropout = False

	def forward_one_step(self, x, test):
		chain = [x]

		for i in range(self.n_layers):
			output = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_dropout and i != self.n_layers - 1:
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
		self.hidden_activation_function = "elu"
		self.output_activation_function = None
		self.apply_dropout = False

	def forward_one_step(self, x, test):
		f = activations[self.hidden_activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if i == self.n_layers - 1:
				if self.output_activation_function:
					output = activations[self.output_activation_function](u)
				else:
					output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

def _sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = 'GradientClipping'

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(_sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			norm = 1
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class AttentiveReader:
	name = "ar"
	def __init__(self):
		config.check()
		wscale = 1.0

		forward_lstm_attributes = {}
		forward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(forward_lstm_units):
			forward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		self.forward_lstm = StackedLSTM(**forward_lstm_attributes)
		self.forward_lstm.n_layers = len(forward_lstm_units)
		self.forward_lstm.apply_dropout = config.bi_lstm_apply_dropout

		backward_lstm_attributes = {}
		backward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(backward_lstm_units):
			backward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		self.backward_lstm = StackedLSTM(**backward_lstm_attributes)
		self.backward_lstm.n_layers = len(backward_lstm_units)
		self.backward_lstm.apply_dropout = config.bi_lstm_apply_dropout

		self.char_embed = L.EmbedID(config.n_vocab, config.ndim_char_embed)

		self.f_ym = L.Linear(config.bi_lstm_units[-1] * 2, config.ndim_m, nobias=True)
		self.f_um = L.Linear(config.bi_lstm_units[-1] * 2, config.ndim_m, nobias=True)

		attention_fc_attributes = {}
		attention_fc_units = zip(config.attention_fc_units[:-1], config.attention_fc_units[1:])
		for i, (n_in, n_out) in enumerate(attention_fc_units):
			attention_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		self.attention_fc = FullyConnectedNetwork(**attention_fc_attributes)
		self.attention_fc.n_layers = len(attention_fc_units)
		self.attention_fc.hidden_activation_function = config.attention_fc_hidden_activation_function
		self.attention_fc.output_activation_function = config.attention_fc_output_activation_function
		self.attention_fc.apply_dropout = config.attention_fc_apply_dropout
			
		self.f_rg = L.Linear(config.bi_lstm_units[-1] * 2, config.ndim_g, nobias=True)
		self.f_ug = L.Linear(config.bi_lstm_units[-1] * 2, config.ndim_g, nobias=True)

		reader_fc_attributes = {}
		reader_fc_units = zip(config.reader_fc_units[:-1], config.reader_fc_units[1:])
		for i, (n_in, n_out) in enumerate(reader_fc_units):
			reader_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		self.reader_fc = FullyConnectedNetwork(**reader_fc_attributes)
		self.reader_fc.n_layers = len(reader_fc_units)
		self.reader_fc.hidden_activation_function = config.reader_fc_hidden_activation_function
		self.reader_fc.output_activation_function = config.reader_fc_output_activation_function
		self.reader_fc.apply_dropout = config.attention_fc_apply_dropout

		if config.use_gpu:
			self.forward_lstm.to_gpu()
			self.backward_lstm.to_gpu()
			self.char_embed.to_gpu()
			self.attention_fc.to_gpu()
			self.reader_fc.to_gpu()
			self.f_ym.to_gpu()
			self.f_um.to_gpu()
			self.f_rg.to_gpu()
			self.f_ug.to_gpu()

		self.optimizer_char_embed = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_char_embed.setup(self.char_embed)
		self.optimizer_char_embed.add_hook(GradientClipping(10.0))

		self.optimizer_forward_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_forward_lstm.setup(self.forward_lstm)
		self.optimizer_forward_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_backward_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_backward_lstm.setup(self.backward_lstm)
		self.optimizer_backward_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_f_um = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_um.setup(self.f_um)
		self.optimizer_f_um.add_hook(GradientClipping(10.0))
		
		self.optimizer_f_ym = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_ym.setup(self.f_ym)
		self.optimizer_f_ym.add_hook(GradientClipping(10.0))
		
		self.optimizer_attention_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_attention_fc.setup(self.attention_fc)
		self.optimizer_attention_fc.add_hook(GradientClipping(10.0))
		
		self.optimizer_f_rg = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_rg.setup(self.f_rg)
		self.optimizer_f_rg.add_hook(GradientClipping(10.0))
		
		self.optimizer_f_ug = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_ug.setup(self.f_ug)
		self.optimizer_f_ug.add_hook(GradientClipping(10.0))
		
		self.optimizer_reader_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_reader_fc.setup(self.reader_fc)
		self.optimizer_reader_fc.add_hook(GradientClipping(10.0))

	def encode(self, x_seq, test=False):
		self.reset_state()
		xp = self.xp
		forward_context = []
		backward_context = []
		if len(x_seq) < 1:
			return None, None
		for char in x_seq:
			char = Variable(xp.array([char], dtype=xp.int32))
			embed = self.char_embed(char)
			y = self.forward_lstm(embed, test=test)
			forward_context.append(y)

		for char in x_seq[::-1]:
			char = Variable(xp.array([char], dtype=xp.int32))
			embed = self.char_embed(char)
			y = self.backward_lstm(embed, test=test)
			backward_context.append(y)

		length = len(x_seq)
		context = []
		for t in xrange(length):
			yd_t = concat_variables(forward_context[t], backward_context[length - t - 1])
			context.append(yd_t)
		encode = concat_variables(forward_context[-1], backward_context[-1])
		return context, encode

	def attend(self, context, encode, test=False):
		length = len(context)
		attention_sum = 0
		weights = []
		for t in xrange(length):
			yd_t = context[t]
			if encode is None:
				m_t = F.tanh(self.f_ym(yd_t))
			else:
				m_t = F.tanh(self.f_ym(yd_t) + self.f_um(encode))
			s_t = F.exp(self.attention_fc(m_t, test=False))
			weights.append(s_t)
			attention_sum += s_t
		return weights, attention_sum

	@property
	def xp(self):
		return np if self.char_embed._cpu else cuda.cupy

	def reset_state(self):
		self.forward_lstm.reset_state()
		self.backward_lstm.reset_state()

	def forward_one_step(self, x_seq, pos, test=True, concat_weight=True):
		self.reset_state()
		xp = self.xp
		x_seq = xp.asanyarray(x_seq)
		length = len(x_seq)
		if length < 1:
			if concat_weight:
				return None, None
			else:
				return None, None, None, None
		sum_loss = 0
		target_char = x_seq[pos]
		former = None
		latter = None
		attention_sum = 0

		if pos == 0:
			latter = x_seq[1:]
		elif pos == length - 1:
			former = x_seq[:pos]
		else:
			former = x_seq[:pos]
			latter = x_seq[pos + 1:]

		former_context = None
		latter_context = None
		former_attention_weight = None
		latter_attention_weight = None

		if former is not None:
			former_context, former_encode = self.encode(former, test=test)
			former_attention_weight, former_attention_sum = self.attend(former_context, former_encode, test=test)
			attention_sum += former_attention_sum
		if latter is not None:
			latter_context, latter_encode = self.encode(latter, test=test)
			latter_attention_weight, latter_attention_sum = self.attend(latter_context, latter_encode, test=test)
			attention_sum += latter_attention_sum

		representation = 0

		if former_context is not None:
			for t in xrange(len(former_context)):
				representation += apply_attention(former_context[t], former_attention_weight[t] / attention_sum)
		if latter_context is not None:
			for t in xrange(len(latter_context)):
				representation += apply_attention(latter_context[t], latter_attention_weight[t] / attention_sum)

		g = self.f_rg(representation)
		predicted_char_embed = self.reader_fc(g)

		if concat_weight:
			batchsize = 1
			weight = xp.zeros((batchsize, length), dtype=xp.float32)
			index = 0
			if former_attention_weight is not None:
				for i in xrange(len(former_attention_weight)):
					index = i
					weight[:, i] = former_attention_weight[i].data
				index += 1
			if latter_attention_weight is not None:
				for i in xrange(len(latter_attention_weight)):
					weight[:, index + i + 1] = latter_attention_weight[i].data
			weight /= attention_sum.data
			if xp is cuda.cupy:
				weight = cuda.to_cpu(weight)
			return weight, predicted_char_embed
		else:
			return former_attention_weight, latter_attention_weight, attention_sum, predicted_char_embed

	def train(self, x_seq, test=False):
		xp = self.xp
		if len(x_seq) < 2:
			return 0
		x_seq = xp.asanyarray(x_seq)
		sum_loss = 0
		for pos in xrange(len(x_seq)):
			target_char = x_seq[pos]
			target_char_embed = self.char_embed(Variable(xp.asarray([target_char], dtype=xp.int32)))
			_, predicted_char_embed = self.forward_one_step(x_seq, pos, test=test)
			if predicted_char_embed is None:
				continue
			loss = F.mean_squared_error(predicted_char_embed, target_char_embed)
			sum_loss += loss

		self.zero_grads()
		sum_loss.backward()
		self.update()

		if xp is cuda.cupy:
			sum_loss.to_cpu()

		return sum_loss.data

	def zero_grads(self):
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.optimizer.GradientMethod):
				prop.zero_grads()

	def update(self):
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.optimizer.GradientMethod):
				prop.update()

	def inverse_embed(self, embed):
		if self.xp is cuda.cupy:
			embed = cuda.to_gpu(embed)
		onehot = self.char_embed.W.data.dot(embed.reshape(-1, 1))
		if self.xp is cuda.cupy:
			onehot = cuda.to_cpu(onehot)
		return onehot

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, L.Linear) or isinstance(prop, L.EmbedID) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/%s-%s.hdf5" % (self.name, attr)
				if os.path.isfile(filename):
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
			if isinstance(prop, chainer.Chain) or isinstance(prop, L.Linear) or isinstance(prop, L.EmbedID) or isinstance(prop, chainer.optimizer.GradientMethod):
				serializers.save_hdf5(dir + "/%s-%s.hdf5" % (self.name, attr), prop)
		print "model saved."

class OneDirectionAttentiveReader(AttentiveReader):
	name = "odar"
	def __init__(self):
		config.check()
		wscale = 1.0

		forward_lstm_attributes = {}
		forward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(forward_lstm_units):
			forward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		self.forward_lstm = StackedLSTM(**forward_lstm_attributes)
		self.forward_lstm.n_layers = len(forward_lstm_units)
		self.forward_lstm.apply_dropout = config.bi_lstm_apply_dropout

		backward_lstm_attributes = {}
		backward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

		for i, (n_in, n_out) in enumerate(backward_lstm_units):
			backward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

		self.backward_lstm = StackedLSTM(**backward_lstm_attributes)
		self.backward_lstm.n_layers = len(backward_lstm_units)
		self.backward_lstm.apply_dropout = config.bi_lstm_apply_dropout

		self.char_embed = L.EmbedID(config.n_vocab, config.ndim_char_embed)

		self.f_ym = L.Linear(config.bi_lstm_units[-1], config.ndim_m, nobias=True)
		self.f_um = L.Linear(config.bi_lstm_units[-1], config.ndim_m, nobias=True)

		attention_fc_attributes = {}
		attention_fc_units = zip(config.attention_fc_units[:-1], config.attention_fc_units[1:])
		for i, (n_in, n_out) in enumerate(attention_fc_units):
			attention_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		self.attention_fc = FullyConnectedNetwork(**attention_fc_attributes)
		self.attention_fc.n_layers = len(attention_fc_units)
		self.attention_fc.hidden_activation_function = config.attention_fc_hidden_activation_function
		self.attention_fc.output_activation_function = config.attention_fc_output_activation_function
		self.attention_fc.apply_dropout = config.attention_fc_apply_dropout
			
		self.f_rg = L.Linear(config.bi_lstm_units[-1], config.ndim_g, nobias=True)
		self.f_ug = L.Linear(config.bi_lstm_units[-1], config.ndim_g, nobias=True)

		reader_fc_attributes = {}
		reader_fc_units = zip(config.reader_fc_units[:-1], config.reader_fc_units[1:])
		for i, (n_in, n_out) in enumerate(reader_fc_units):
			reader_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		self.reader_fc = FullyConnectedNetwork(**reader_fc_attributes)
		self.reader_fc.n_layers = len(reader_fc_units)
		self.reader_fc.hidden_activation_function = config.reader_fc_hidden_activation_function
		self.reader_fc.output_activation_function = config.reader_fc_output_activation_function
		self.reader_fc.apply_dropout = config.attention_fc_apply_dropout

		if config.use_gpu:
			self.forward_lstm.to_gpu()
			self.backward_lstm.to_gpu()
			self.char_embed.to_gpu()
			self.attention_fc.to_gpu()
			self.reader_fc.to_gpu()
			self.f_ym.to_gpu()
			self.f_um.to_gpu()
			self.f_rg.to_gpu()
			self.f_ug.to_gpu()

		self.optimizer_char_embed = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_char_embed.setup(self.char_embed)
		self.optimizer_char_embed.add_hook(GradientClipping(10.0))

		self.optimizer_forward_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_forward_lstm.setup(self.forward_lstm)
		self.optimizer_forward_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_backward_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_backward_lstm.setup(self.backward_lstm)
		self.optimizer_backward_lstm.add_hook(GradientClipping(10.0))

		self.optimizer_f_um = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_um.setup(self.f_um)
		self.optimizer_f_um.add_hook(GradientClipping(10.0))
		
		self.optimizer_f_ym = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_ym.setup(self.f_ym)
		self.optimizer_f_ym.add_hook(GradientClipping(10.0))
		
		self.optimizer_attention_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_attention_fc.setup(self.attention_fc)
		self.optimizer_attention_fc.add_hook(GradientClipping(10.0))
		
		self.optimizer_f_rg = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_rg.setup(self.f_rg)
		self.optimizer_f_rg.add_hook(GradientClipping(10.0))
		
		self.optimizer_f_ug = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_ug.setup(self.f_ug)
		self.optimizer_f_ug.add_hook(GradientClipping(10.0))
		
		self.optimizer_reader_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_reader_fc.setup(self.reader_fc)
		self.optimizer_reader_fc.add_hook(GradientClipping(10.0))

	def encode_forward(self, x_seq, test=False):
		self.reset_state()
		xp = self.xp
		forward_context = []
		for char in x_seq:
			char = Variable(xp.array([char], dtype=xp.int32))
			embed = self.char_embed(char)
			y = self.forward_lstm(embed, test=test)
			forward_context.append(y)
		return forward_context, forward_context[-1]

	def encode_backward(self, x_seq, test=False):
		self.reset_state()
		xp = self.xp
		backward_context = []
		for char in x_seq[::-1]:
			char = Variable(xp.array([char], dtype=xp.int32))
			embed = self.char_embed(char)
			y = self.backward_lstm(embed, test=test)
			backward_context.append(y)
		return backward_context, backward_context[-1]

	def attend(self, context, encode, test=False):
		length = len(context)
		attention_sum = 0
		weights = []
		for t in xrange(length):
			yd_t = context[t]
			if encode is None:
				m_t = F.tanh(self.f_ym(yd_t))
			else:
				m_t = F.tanh(self.f_ym(yd_t) + self.f_um(encode))
			s_t = F.exp(self.attention_fc(m_t, test=False))
			weights.append(s_t)
			attention_sum += s_t
		return weights, attention_sum

	def forward_one_step(self, x_seq, pos, test=True, concat_weight=True):
		self.reset_state()
		xp = self.xp
		x_seq = xp.asanyarray(x_seq)
		length = len(x_seq)
		if length < 1:
			if concat_weight:
				return None, None
			else:
				return None, None, None, None
		sum_loss = 0
		target_char = x_seq[pos]
		former = None
		latter = None
		attention_sum = 0

		if pos == 0:
			latter = x_seq[1:]
		elif pos == length - 1:
			former = x_seq[:pos]
		else:
			former = x_seq[:pos]
			latter = x_seq[pos + 1:]

		former_context = None
		latter_context = None
		former_attention_weight = None
		latter_attention_weight = None

		if former is not None:
			former_context, former_encode = self.encode_backward(former, test=test)
			former_attention_weight, former_attention_sum = self.attend(former_context, None, test=test)
			attention_sum += former_attention_sum
		if latter is not None:
			latter_context, latter_encode = self.encode_forward(latter, test=test)
			latter_attention_weight, latter_attention_sum = self.attend(latter_context, None, test=test)
			attention_sum += latter_attention_sum

		representation = 0

		if former_context is not None:
			for t in xrange(len(former_context)):
				representation += apply_attention(former_context[t], former_attention_weight[t] / attention_sum)
		if latter_context is not None:
			for t in xrange(len(latter_context)):
				representation += apply_attention(latter_context[t], latter_attention_weight[t] / attention_sum)

		g = self.f_rg(representation)
		predicted_char_embed = self.reader_fc(g)

		if concat_weight:
			batchsize = 1
			weight = xp.zeros((batchsize, length), dtype=xp.float32)
			index = 0
			if former_attention_weight is not None:
				f_length = len(former_attention_weight)
				for i in xrange(f_length):
					index = i
					weight[:, f_length - i - 1] = former_attention_weight[i].data
				index += 1
			if latter_attention_weight is not None:
				for i in xrange(len(latter_attention_weight)):
					weight[:, index + i + 1] = latter_attention_weight[i].data
			weight /= attention_sum.data
			if xp is cuda.cupy:
				weight = cuda.to_cpu(weight)
			return weight, predicted_char_embed
		else:
			return former_attention_weight, latter_attention_weight, attention_sum, predicted_char_embed

class Concat(function.Function):
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		a_type, b_type = in_types

		type_check.expect(
			a_type.dtype == np.float32,
			b_type.dtype == np.float32,
			a_type.ndim == 2,
			b_type.ndim == 2,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		v_a, v_b = inputs
		n_batch = v_a.shape[0]
		output = xp.empty((n_batch, v_a.shape[1] + v_b.shape[1]), dtype=xp.float32)
		output[:,:v_a.shape[1]] = v_a
		output[:,v_a.shape[1]:] = v_b
		return output,

	def backward(self, inputs, grad_outputs):
		v_a, v_b = inputs
		return grad_outputs[0][:,:v_a.shape[1]], grad_outputs[0][:,v_a.shape[1]:]

def concat_variables(v_a, v_b):
	return Concat()(v_a, v_b)

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
		output = context * weight
		return output,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(inputs[0])
		context, weight = inputs
		return grad_outputs[0] * weight, xp.sum(grad_outputs[0] * context, axis=1).reshape(-1, 1)

def apply_attention(context, weight):
	return Attention()(context, weight)

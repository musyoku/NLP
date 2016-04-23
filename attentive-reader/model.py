# -*- coding: utf-8 -*-
import os, time, re
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

class AttentiveReader:
	def __init__(self, char_embed, forward_lstm, backward_lstm, f_um, f_ym, attention_fc, f_rg, f_ug, reader_fc):
		self.char_embed = char_embed
		self.optimizer_char_embed = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_char_embed.setup(self.char_embed)
		self.optimizer_char_embed.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.forward_lstm = forward_lstm
		self.optimizer_forward_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_forward_lstm.setup(self.forward_lstm)
		self.optimizer_forward_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.backward_lstm = backward_lstm
		self.optimizer_backward_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_backward_lstm.setup(self.backward_lstm)
		self.optimizer_backward_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.f_um = f_um
		self.optimizer_f_um = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_um.setup(self.f_um)
		self.optimizer_f_um.add_hook(chainer.optimizer.GradientClipping(10.0))
		
		self.f_ym = f_ym
		self.optimizer_f_ym = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_ym.setup(self.f_ym)
		self.optimizer_f_ym.add_hook(chainer.optimizer.GradientClipping(10.0))
		
		self.attention_fc = attention_fc
		self.optimizer_attention_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_attention_fc.setup(self.attention_fc)
		self.optimizer_attention_fc.add_hook(chainer.optimizer.GradientClipping(10.0))
		
		self.f_rg = f_rg
		self.optimizer_f_rg = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_rg.setup(self.f_rg)
		self.optimizer_f_rg.add_hook(chainer.optimizer.GradientClipping(10.0))
		
		self.f_ug = f_ug
		self.optimizer_f_ug = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_f_ug.setup(self.f_ug)
		self.optimizer_f_ug.add_hook(chainer.optimizer.GradientClipping(10.0))
		
		self.reader_fc = reader_fc
		self.optimizer_reader_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_reader_fc.setup(self.reader_fc)
		self.optimizer_reader_fc.add_hook(chainer.optimizer.GradientClipping(10.0))

	def encode(self, x_seq, test=False):
		self.reset_state()
		xp = self.xp
		forward_context = []
		backward_context = []
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
		return forward_context, backward_context

	def attend(self, forward_context, backward_context, test=False):
		u = concat_variables(forward_context[-1], backward_context[-1])
		length = len(forward_context)
		attention_sum = 0
		for t in xrange(length):
			yd_t = concat_variables(forward_context[t], backward_context[length - t - 1])
			m_t = F.tanh(self.f_ym(yd_t) + self.f_um(u))
			s_t = self.attention_fc(m_t, test=False)
			attention_sum += s_t
		print attention_sum.data

	@property
	def xp(self):
		return np if self.char_embed._cpu else cuda.cupy

	def reset_state(self):
		self.forward_lstm.reset_state()
		self.backward_lstm.reset_state()

	def train(self, x_seq, test=False):
		length = len(x_seq)
		for t in xrange(length - 1):
			target = x_seq[t]
			if t == 0:
				latter = x_seq[1:]
				print latter
			elif t == length - 2:
				former = x_seq[:t-1]
				print former
			else:
				former = x_seq[:t]
				latter = x_seq[t + 1:]
				print former, latter
		forward_context, backward_context = self.encode(x_seq, test=test)
		self.attend(forward_context, backward_context, test=test)

	def load(self, dir=None, name="ar"):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, L.Linear) or isinstance(prop, L.EmbedID) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/%s.hdf5" % attr
				if os.path.isfile(filename):
					serializers.load_hdf5(filename, prop)
					print attr, "loaded."

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
				serializers.save_hdf5(dir + "/%s.hdf5" % attr, prop)
				print attr, "saved."

def build():
	config.check()
	wscale = 1.0

	forward_lstm_attributes = {}
	forward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

	for i, (n_in, n_out) in enumerate(forward_lstm_units):
		forward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

	forward_lstm = StackedLSTM(**forward_lstm_attributes)
	forward_lstm.n_layers = len(forward_lstm_units)
	forward_lstm.apply_dropout = config.bi_lstm_apply_dropout

	backward_lstm_attributes = {}
	backward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

	for i, (n_in, n_out) in enumerate(backward_lstm_units):
		backward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)

	backward_lstm = StackedLSTM(**backward_lstm_attributes)
	backward_lstm.n_layers = len(backward_lstm_units)
	backward_lstm.apply_dropout = config.bi_lstm_apply_dropout

	char_embed = L.EmbedID(config.n_vocab, config.ndim_char_embed)

	f_ym = L.Linear(config.bi_lstm_units[-1] * 2, config.ndim_m, nobias=True)
	f_um = L.Linear(config.bi_lstm_units[-1] * 2, config.ndim_m, nobias=True)

	attention_fc_attributes = {}
	attention_fc_units = zip(config.attention_fc_units[:-1], config.attention_fc_units[1:])
	for i, (n_in, n_out) in enumerate(attention_fc_units):
		attention_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
	attention_fc = FullyConnectedNetwork(**attention_fc_attributes)
	attention_fc.n_layers = len(attention_fc_units)
	attention_fc.hidden_activation_function = config.attention_fc_hidden_activation_function
	attention_fc.output_activation_function = config.attention_fc_output_activation_function
	attention_fc.apply_dropout = config.attention_fc_apply_dropout
		
	f_rg = L.Linear(config.ndim_m, config.ndim_g, nobias=True)
	f_ug = L.Linear(config.ndim_m, config.ndim_g, nobias=True)

	reader_fc_attributes = {}
	reader_fc_units = zip(config.reader_fc_units[:-1], config.reader_fc_units[1:])
	for i, (n_in, n_out) in enumerate(reader_fc_units):
		reader_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
	reader_fc = FullyConnectedNetwork(**reader_fc_attributes)
	reader_fc.n_layers = len(reader_fc_units)
	reader_fc.hidden_activation_function = config.reader_fc_hidden_activation_function
	reader_fc.output_activation_function = config.reader_fc_output_activation_function
	reader_fc.apply_dropout = config.attention_fc_apply_dropout

	if config.use_gpu:
		forward_lstm.to_gpu()
		backward_lstm.to_gpu()
		char_embed.to_gpu()
		attention_fc.to_gpu()
		reader_fc.to_gpu()
		f_ym.to_gpu()
		f_um.to_gpu()
		f_rg.to_gpu()
		f_ug.to_gpu()

	return AttentiveReader(char_embed, forward_lstm, backward_lstm, f_um, f_ym, attention_fc, f_rg, f_ug, reader_fc)

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
		n_batch = context.shape[0]
		output = xp.empty((n_batch, context.shape[1] + weight.shape[1]), dtype=xp.float32)
		output[:,:context.shape[1]] = context
		output[:,context.shape[1]:] = weight
		return output,

	def backward(self, inputs, grad_outputs):
		context, weight = inputs
		return grad_outputs[0][:,:context.shape[1]], grad_outputs[0][:,context.shape[1]:]

def apply_attention(context, weight):
	return Concat()(context, weight)

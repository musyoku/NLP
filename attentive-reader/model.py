# -*- coding: utf-8 -*-
import os, time
import numpy as np
import chainer
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from config import config
from activations import activations

class LSTM(chainer.Chain):
	def __init__(self, **layers):
		super(LSTM, self).__init__(**layers)
		self.n_layers = 0
		self.activation_function = None
		self.apply_dropout = False
		self.apply_batchnorm = False
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			u = chain[-1]
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = getattr(self, "layer_%i" % i)(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		return chain[-1]

	def reset_state(self):
		for i in range(self.n_layers):
			getattr(self, "layer_%i" % i).reset_state()

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class Model:
	def __init__(self, forward_lstm, backward_lstm, f_um, f_ym, f_ms, f_rg, f_ug):
		self.encoder_embed = encoder_embed
		self.optimizer_encoder_embed = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_encoder_embed.setup(self.encoder_embed)
		self.optimizer_encoder_embed.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.encoder_lstm = encoder_lstm
		self.optimizer_encoder_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_encoder_lstm.setup(self.encoder_lstm)
		self.optimizer_encoder_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.decoder_lstm = decoder_lstm
		self.optimizer_decoder_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_decoder_lstm.setup(self.decoder_lstm)
		self.optimizer_decoder_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.decoder_fc = decoder_fc
		self.optimizer_decoder_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_decoder_fc.setup(self.decoder_fc)
		self.optimizer_decoder_fc.add_hook(chainer.optimizer.GradientClipping(10.0))

	# Inputs:	Numpy / CuPy
	# Returns:	Variable
	def encode(self, x_seq_batch, test=False):
		self.encoder_lstm.reset_state()
		xp = self.xp
		x_seq_batch = x_seq_batch.T
		for i, cur_x in enumerate(x_seq_batch):
			cur_x[cur_x == -1] = 0
			cur_x = Variable(xp.asanyarray(cur_x, dtype=np.int32))
			embed = self.encoder_embed(cur_x)
			output = self.encoder_lstm(embed, test=test)
		return output

	# Inputs:	Variable, Variable
	# Returns:	Confidence of each word ID
	# Note:		decoderのLSTMの内部状態は適当にリセットしておいてください
	def decode_one_step(self, summary, prev_y, test=False, softmax=True):
		input = append_variable(summary, prev_y)
		h = self.decoder_lstm(input, test=test)
		output = self.decoder_fc(h, test=test)
		if softmax:
			output = F.softmax(output)
		return output

	# Inputs:	Numpy / CuPy
	# Outputs:	Numpy / CuPy
	# Note:		sampling_yがTrueだと出力されるIDはsoftmax出力からサンプリングします。
	#			そうでない場合は確信度が最も高いものを取ります。（訓練データを再生するだけの意味のないものになるかもしれません）
	# 			decoderのLSTMの内部状態は適当にリセットしておいてください
	def decode(self, x_seq, test=False, limit=1000, sampling_y=True):
		xp = self.xp
		summary = self.encode(x_seq.reshape(1, -1), test=test)
		y_seq = None
		ids = np.arange(config.n_vocab, dtype=np.uint8)
		prev_y = Variable(xp.zeros((1, config.n_vocab), dtype=xp.float32))
		for t in xrange(limit):
			if sampling_y:
				distribution = self.decode_one_step(summary, prev_y, test=test, softmax=True)
				if xp is cuda.cupy:
					distribution.to_cpu()
				y = np.random.choice(ids, 1, p=distribution.data[0])
			else:
				confidence = self.decode_one_step(summary, prev_y, test=test, softmax=False)
				if xp is cuda.cupy:
					confidence.to_cpu()
				y = np.argmax(confidence.data[0], axis=0)
				y = np.asarray([y], dtype=np.uint8)
			if y_seq is None:
				y_seq = y
			else:
				y_seq = np.append(y_seq, y)
			y = y[0]
			if y == 0:
				break
			prev_y = xp.zeros((1, config.n_vocab), dtype=xp.float32)
			prev_y[0, y] = 1
			prev_y = Variable(prev_y)
		return y_seq

	@property
	def xp(self):
		return np if self.encoder_lstm.layer_0._cpu else cuda.cupy

	def reset_state(self):
		self.encoder_lstm.reset_state()
		self.decoder_lstm.reset_state()

	def learn(self, source_seq_batch, target_seq_batch, test=False):
		self.reset_state()
		xp = self.xp
		n_batch = source_seq_batch.shape[0]
		sum_loss = 0
		summary = self.encode(source_seq_batch, test=test)
		target_seq_batch = target_seq_batch.T
		prev_y_onehot = Variable(xp.zeros((n_batch, config.n_vocab), dtype=xp.float32))
		for i, target_y_ids in enumerate(target_seq_batch):
			confidence_y = self.decode_one_step(summary, prev_y_onehot, test=test, softmax=False)
			target_y = Variable(target_y_ids)
			if xp is cuda.cupy:
				target_y.to_gpu()
			loss = F.softmax_cross_entropy(confidence_y, target_y)
			sum_loss += loss
			# CuPy does not support advanced indexing
			prev_y_ids = target_y_ids
			prev_y_ids[prev_y_ids == -1] = 0
			prev_y_onehot = np.zeros(prev_y_onehot.data.shape, dtype=np.float32)
			prev_y_onehot[np.arange(n_batch), prev_y_ids.astype(np.int32)] = 1
			prev_y_onehot = Variable(prev_y_onehot)
			if xp is cuda.cupy:
				prev_y_onehot.to_gpu()

		self.optimizer_encoder_embed.zero_grads()
		self.optimizer_encoder_lstm.zero_grads()
		self.optimizer_decoder_lstm.zero_grads()
		self.optimizer_decoder_fc.zero_grads()
		sum_loss.backward()
		self.optimizer_encoder_embed.update()
		self.optimizer_encoder_lstm.update()
		self.optimizer_decoder_lstm.update()
		self.optimizer_decoder_fc.update()
		return sum_loss.data

	def load(self, dir=None, name="encoder_lstm"):
		if dir is None:
			raise Exception()
		filename = dir + "/%s_decoder_fc.model" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.decoder_fc)
			print filename, "loaded."
		filename = dir + "/%s_decoder_lstm.model" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.decoder_lstm)
			print filename, "loaded."
		filename = dir + "/%s_encoder_embed.model" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.encoder_embed)
			print filename, "loaded."
		filename = dir + "/%s_encoder_lstm.model" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.encoder_lstm)
			print filename, "loaded."
		filename = dir + "/%s_decoder_fc.optimizer" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_decoder_fc)
			print filename, "loaded."
		filename = dir + "/%s_decoder_lstm.optimizer" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_decoder_lstm)
			print filename, "loaded."
		filename = dir + "/%s_encoder_embed.optimizer" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_encoder_embed)
			print filename, "loaded."
		filename = dir + "/%s_encoder_lstm.optimizer" % name
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_encoder_lstm)
			print filename, "loaded."

	def save(self, dir=None, name="encoder_lstm"):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		serializers.save_hdf5(dir + "/%s_decoder_fc.model" % name, self.decoder_fc)
		serializers.save_hdf5(dir + "/%s_decoder_lstm.model" % name, self.decoder_lstm)
		serializers.save_hdf5(dir + "/%s_encoder_embed.model" % name, self.encoder_embed)
		serializers.save_hdf5(dir + "/%s_encoder_lstm.model" % name, self.encoder_lstm)
		print "model saved."
		serializers.save_hdf5(dir + "/%s_decoder_fc.optimizer" % name, self.optimizer_decoder_fc)
		serializers.save_hdf5(dir + "/%s_decoder_lstm.optimizer" % name, self.optimizer_decoder_lstm)
		serializers.save_hdf5(dir + "/%s_encoder_embed.optimizer" % name, self.optimizer_encoder_embed)
		serializers.save_hdf5(dir + "/%s_encoder_lstm.optimizer" % name, self.optimizer_encoder_lstm)
		print "optimizer saved."


def build():
	config.check()
	wscale = 1.0

	forward_lstm_attributes = {}
	forward_lstm_units = zip(config.bi_lstm_units[:-1], config.bi_lstm_units[1:])

	for i, (n_in, n_out) in enumerate(forward_lstm_units):
		forward_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)
		forward_lstm_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

	forward_lstm = LSTM(**forward_lstm_attributes)
	forward_lstm.n_layers = len(forward_lstm_units) - 1
	forward_lstm.apply_batchnorm = config.forward_lstm_apply_batchnorm
	forward_lstm.apply_batchnorm_to_input = config.forward_lstm_apply_batchnorm_to_input
	forward_lstm.apply_dropout = config.forward_lstm_apply_dropout
	if config.use_gpu:
		forward_lstm.to_gpu()

	enc_embed = L.EmbedID(config.n_vocab, config.enc_lstm_units[0])
	if config.use_gpu:
		enc_embed.to_gpu()

	dec_lstm_attributes = {}
	dec_lstm_units = zip(config.dec_lstm_units[:-1], config.dec_lstm_units[1:])

	for i, (n_in, n_out) in enumerate(dec_lstm_units):
		dec_lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out)
		dec_lstm_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

	dec_lstm = LSTM(**dec_lstm_attributes)
	dec_lstm.n_layers = len(dec_lstm_units)
	dec_lstm.apply_batchnorm = config.dec_lstm_apply_batchnorm
	dec_lstm.apply_batchnorm_to_input = config.dec_lstm_apply_batchnorm_to_input
	dec_lstm.apply_dropout = config.dec_lstm_apply_dropout
	if config.use_gpu:
		dec_lstm.to_gpu()

	dec_fc_attributes = {}
	dec_fc_units = zip(config.dec_fc_units[:-1], config.dec_fc_units[1:])
	dec_fc_units += [(config.dec_fc_units[-1], config.n_vocab)]

	for i, (n_in, n_out) in enumerate(dec_fc_units):
		dec_fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		dec_fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

	dec_fc = FullyConnectedNetwork(**dec_fc_attributes)
	dec_fc.n_hidden_layers = len(dec_fc_units) - 1
	dec_fc.activation_function = config.dec_fc_activation_function
	dec_fc.apply_batchnorm_to_input = config.dec_fc_apply_batchnorm_to_input
	dec_fc.apply_batchnorm_to_output = config.dec_fc_apply_batchnorm_to_output
	dec_fc.apply_batchnorm = config.dec_fc_apply_batchnorm
	dec_fc.apply_dropout = config.dec_fc_apply_dropout
	if config.use_gpu:
		dec_fc.to_gpu()

	return Model(enc_embed, enc_lstm, dec_lstm, dec_fc)

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
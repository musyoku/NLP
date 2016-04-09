# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import model

class LSTM(chainer.Chain):
	def __init__(self, **layers):
		super(LSTM, self).__init__(**layers)
		self.n_layers = 0
		self.activation_function = "elu"
		self.apply_dropout = False
		self.apply_batchnorm = False
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
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
		self.n_hidden_layers = 0
		self.activation_function = "elu"
		self.apply_dropout = False
		self.apply_batchnorm = False
		self.apply_batchnorm_to_input = False
		self.apply_batchnorm_to_output = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % self.n_hidden_layers)(chain[-1])
		if self.apply_batchnorm_to_output:
			u = getattr(self, "batchnorm_%i" % self.n_hidden_layers)(u, test=test)
		chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class Model:
	def __init__(self, lstm, fc):
		self.lstm = lstm
		self.optimizer_lstm = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_lstm.setup(self.lstm)
		self.optimizer_lstm.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.fc = fc
		self.optimizer_fc = optimizers.Adam(alpha=config.learning_rate, beta1=config.gradient_momentum)
		self.optimizer_fc.setup(self.fc)
		self.optimizer_fc.add_hook(chainer.optimizer.GradientClipping(10.0))

	def __call__(self, x, test=False):
		output = self.lstm(x, test=test)
		output = self.fc(output, test=test)
		return output

	def learn(self, x, target, test=False):
		output = self(x, test=False)
		loss = F.mean_squared_error(output, target)
		self.optimizer_lstm.zero_grads()
		self.optimizer_fc.zero_grads()
		loss.backward()
		self.optimizer_lstm.update()
		self.optimizer_fc.update()

def build():
	config.check()
	wscale = 1.0

	lstm_attributes = {}
	lstm_units = zip(config.lstm_units[:-1], config.lstm_units[1:])

	for i, (n_in, n_out) in enumerate(lstm_units):
		lstm_attributes["layer_%i" % i] = L.LSTM(n_in, n_out, wscale=wscale)
		lstm_attributes["batchnorm_%i" % i] = BatchNormalization(n_out)

	lstm = LSTM(**lstm_attributes)
	lstm.n_layers = len(lstm_units)
	lstm.activation_function = config.q_lstm_activation_function
	lstm.apply_batchnorm = config.apply_batchnorm
	lstm.apply_batchnorm_to_input = config.lstm_apply_batchnorm_to_input
	lstm.apply_dropout = config.lstm_apply_dropout
	if config.use_gpu:
		lstm.to_gpu()

	fc_attributes = {}
	fc_units = zip(config.fc_units[:-1], config.fc_units[1:])

	for i, (n_in, n_out) in enumerate(fc_units):
		fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
		fc_attributes["batchnorm_%i" % i] = BatchNormalization(n_out)

	fc = FullyConnectedNetwork(**fc_attributes)
	fc.n_hidden_layers = len(fc_units) - 1
	fc.activation_function = config.q_fc_activation_function
	fc.apply_batchnorm_to_input = config.fc_apply_batchnorm_to_input
	fc.apply_batchnorm_to_output = config.fc_apply_batchnorm_to_output
	fc.apply_batchnorm = config.apply_batchnorm
	fc.apply_dropout = config.fc_apply_dropout
	if config.use_gpu:
		fc.to_gpu()
	return lstm, fc

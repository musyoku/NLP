import numpy

import chainer
from chainer import function
from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import softplus
from chainer.functions.activation import tanh
from chainer.utils import type_check
from chainer import link
from chainer.links.connection import linear
from chainer import variable

class LinearInterpolate(function.Function):

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 3)
		p_type, x_type, y_type = in_types

		type_check.expect(
			p_type.dtype.kind == 'f',
			x_type.dtype == p_type.dtype,
			y_type.dtype == p_type.dtype,
			p_type.shape == x_type.shape,
			p_type.shape == y_type.shape,
		)

	def forward_cpu(self, inputs):
		p, x, y = inputs
		one = p.dtype.type(1)
		return utils.force_array(p * x + (one - p) * y),

	def forward_gpu(self, inputs):
		p, x, y = inputs
		return cuda.elementwise(
			'T p, T x, T y', 'T z',
			'z = p * x + (1 - p) * y',
			'linear_interpolate_fwd',
		)(p, x, y),

	def backward_cpu(self, inputs, grads):
		p, x, y = inputs
		g = grads[0]
		pg = p * g
		return (utils.force_array((x - y) * g),
				utils.force_array(pg),
				utils.force_array(g - pg))

	def backward_gpu(self, inputs, grads):
		p, x, y = inputs
		g = grads[0]
		return cuda.elementwise(
			'T p, T x, T y, T g', 'T gp, T gx, T gy',
			'''
			gp = (x - y) * g;
			gx = g * p;
			gy = g * (1 - p);
			''',
			'linear_interpolate_bwd'
		)(p, x, y, g)


def linear_interpolate(p, x, y):
	return LinearInterpolate()(p, x, y)
	
class HardSigmoid(function.Function):

	"""Hard-sigmoid funciton."""

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 1)
		x_type, = in_types

		type_check.expect(x_type.dtype.kind == 'f')

	def forward_cpu(self, inputs):
		x = inputs[0]
		y = numpy.clip(x * 0.2 + 0.5, 0.0, 1.0)
		return utils.force_array(y).astype(x.dtype),

	def forward_gpu(self, inputs):
		x = inputs[0]
		return cuda.elementwise(
			'T x', 'T y',
			'y = min(1.0, max(0.0, x * 0.2 + 0.5))',
			'hard_sigmoid_fwd'
		)(x),

	def backward_cpu(self, inputs, grads):
		x = inputs[0]
		g = grads[0]
		gx = ((-2.5 < x) & (x < 2.5)) * g * 0.2
		return utils.force_array(gx).astype(x.dtype),

	def backward_gpu(self, inputs, grads):
		x = inputs[0]
		g = grads[0]
		return cuda.elementwise(
			'T x, T g', 'T gx',
			'gx = fabs(x) < 2.5 ? 0.2 * g : 0',
			'hard_sigmoid_bwd'
		)(x, g),


def hard_sigmoid(x):

	return HardSigmoid()(x)

class SGU(link.Chain):

	def __init__(self, in_size, out_size):
		super(SGU, self).__init__(
			W_xh=linear.Linear(in_size, out_size),
			W_zxh=linear.Linear(out_size, out_size),
			W_xz=linear.Linear(in_size, out_size),
			W_hz=linear.Linear(out_size, out_size),
		)

	def __call__(self, h, x):
		x_g = self.W_xh(x)
		z_g = tanh.tanh(self.W_zxh(x_g * h))
		z_out = softplus.softplus(z_g * h)
		z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x) + self.W_hz(h))
		h_t = linear_interpolate(z_t, z_out, h)
		return h_t


class StatefulSGU(SGU):

	def __init__(self, in_size, out_size):
		super(StatefulSGU, self).__init__(in_size, out_size)
		self.state_size = out_size
		self.reset_state()

	def to_cpu(self):
		super(StatefulSGU, self).to_cpu()
		if self.h is not None:
			self.h.to_cpu()

	def to_gpu(self, device=None):
		super(StatefulSGU, self).to_gpu(device)
		if self.h is not None:
			self.h.to_gpu(device)

	def set_state(self, h):
		assert isinstance(h, chainer.Variable)
		h_ = h
		if self.xp is numpy:
			h_.to_cpu()
		else:
			h_.to_gpu()
		self.h = h_

	def reset_state(self):
		self.h = None

	def __call__(self, x):

		if self.h is None:
			xp = cuda.get_array_module(x)
			zero = variable.Variable(xp.zeros_like(x.data))
			z_out = softplus.softplus(zero)
			z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x))
			h_t = z_t * z_out
		else:
			h_t = SGU.__call__(self, self.h, x)

		self.h = h_t
		return h_t


class DSGU(link.Chain):

	def __init__(self, in_size, out_size):
		super(DSGU, self).__init__(
			W_xh=linear.Linear(in_size, out_size),
			W_zxh=linear.Linear(out_size, out_size),
			W_go=linear.Linear(out_size, out_size),
			W_xz=linear.Linear(in_size, out_size),
			W_hz=linear.Linear(out_size, out_size),
		)

	def __call__(self, h, x):
		x_g = self.W_xh(x)
		z_g = tanh.tanh(self.W_zxh(x_g * h))
		z_out = sigmoid.sigmoid(self.W_go(z_g * h))
		z_t = hard_sigmoid(self.W_xz(x) + self.W_hz(h))
		h_t = linear_interpolate(z_t, z_out, h)
		return h_t


class StatefulDSGU(DSGU):

	def __init__(self, in_size, out_size):
		super(StatefulDSGU, self).__init__(in_size, out_size)
		self.state_size = out_size
		self.reset_state()

	def to_cpu(self):
		super(StatefulDSGU, self).to_cpu()
		if self.h is not None:
			self.h.to_cpu()

	def to_gpu(self, device=None):
		super(StatefulDSGU, self).to_gpu(device)
		if self.h is not None:
			self.h.to_gpu(device)

	def set_state(self, h):
		assert isinstance(h, chainer.Variable)
		h_ = h
		if self.xp is numpy:
			h_.to_cpu()
		else:
			h_.to_gpu()
		self.h = h_

	def reset_state(self):
		self.h = None

	def __call__(self, x):

		if self.h is None:
			z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x))
			h_t = z_t * 0.5
		else:
			h_t = DSGU.__call__(self, self.h, x)

		self.h = h_t
		return h_t
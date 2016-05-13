import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import linear
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import reader

class TestAttention(unittest.TestCase):

	def setUp(self):
		self.context = numpy.random.uniform(1, 1, (3, 4)).astype(numpy.float32)
		self.weight = numpy.random.uniform(1, 1, (3, 1)).astype(numpy.float32)
		self.gy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
		self.y = self.context * self.weight

	def check_forward(self, context, weight, y_expect):
		context = chainer.Variable(context)
		weight = chainer.Variable(weight)
		y = reader.apply_attention(context, weight)
		gradient_check.assert_allclose(y_expect, y.data)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.context, self.weight, self.y)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.check_forward(cuda.to_gpu(self.context), cuda.to_gpu(self.weight), cuda.to_gpu(self.context * self.weight))

	def check_backward(self, context, weight, y_grad):
		args = (context, weight)
		gradient_check.check_backward(reader.Attention(), args, y_grad, eps=1e-2)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.context, self.weight, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.context), cuda.to_gpu(self.weight), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)
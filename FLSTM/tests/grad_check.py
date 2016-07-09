# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, gradient_check, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import lstm

xp = cuda.cupy
context = Variable(xp.random.uniform(-1, 1, (4, 3)).astype(xp.float32))
weight = Variable(xp.random.uniform(-1, 1, (4, 2)).astype(xp.float32))
z = Variable(xp.full((4, 2), 10.0).astype(xp.float32))
weight /= z
print context.data
print weight.data
y1 = lstm.apply_attention(context, weight, 0)
print y1.data
y2 = lstm.apply_attention(context, weight, 1)
print y2.data
y_grad = xp.random.uniform(-1.0, 1.0, (4, 3)).astype(xp.float32)
attention = lstm.Attention()
attention.index = 0
gradient_check.check_backward(attention, (context.data, weight.data), y_grad, eps=1e-2)
attention.index = 1
gradient_check.check_backward(attention, (context.data, weight.data), y_grad, eps=1e-2)
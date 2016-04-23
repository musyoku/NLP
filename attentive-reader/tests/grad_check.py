# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, gradient_check, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import model

xp = cuda.cupy
context = xp.random.uniform(-1, 1, (2, 3)).astype(xp.float32)
weight = xp.random.uniform(-1, 1, (2, 1)).astype(xp.float32)
z = xp.full((2, 1), 10.0).astype(xp.float32)
y = model.apply_attention(Variable(context), Variable(weight) / Variable(z))
print y.data
y_grad = xp.random.uniform(-1.0, 1.0, (2, 3)).astype(xp.float32)
gradient_check.check_backward(model.Attention(), (context, weight / z), y_grad, eps=1e-2)
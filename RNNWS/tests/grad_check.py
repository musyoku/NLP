# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, gradient_check, Variable
sys.path.append(os.path.split(os.getcwd())[0])
import model

xp = cuda.cupy
a = xp.full((2, 10), 1).astype(xp.float32)
b = xp.full((2, 10), -1).astype(xp.float32)
y = model.append_variable(Variable(a), Variable(b))
print y.data
y_grad = xp.random.uniform(-1.0, 1.0, (2, 20)).astype(xp.float32)
gradient_check.check_backward(model.Append(), (a, b), y_grad, eps=1e-2)
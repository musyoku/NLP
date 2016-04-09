# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import model
import vocab

data_dir = "text"
data_dir = "debug"
dataset, n_vocab, n_dataset = vocab.load(data_dir)
lm = model.build(n_vocab)

n_epoch = 1000
n_train = 1000

for epoch in xrange(n_epoch):
	sum_loss = 0
	for t in xrange(n_train):
		lm.reset_state()
		k = np.random.randint(0, n_dataset)
		k = 0
		data = dataset[k]
		sum_loss += lm.learn(data)
	print sum_loss / float(n_train)

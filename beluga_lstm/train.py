# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import model
import vocab

data_dir = "text"
model_dir = "model"
dataset, n_vocab, n_dataset = vocab.load(data_dir)
lm = model.build(n_vocab)
lm.load(model_dir)

n_epoch = 1000
n_train = 1000
total_time = 0
for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		lm.reset_state()
		k = np.random.randint(0, n_dataset)
		data = dataset[k]
		sum_loss += lm.learn(data)
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	print "epoch:", epoch, "loss:", sum_loss / float(n_train), "time:", int(elapsed_time / 60), "min", "total_time:", int(total_time / 60), "min"
	lm.save(model_dir)


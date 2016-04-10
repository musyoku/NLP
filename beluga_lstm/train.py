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
batchsize = 32
total_time = 0
for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		batch_array = []
		max_length = 0
		for b in xrange(batchsize):
			k = np.random.randint(0, n_dataset)
			data = dataset[k]
			batch_array.append(data)
			if len(data) > max_length:
				max_length = len(data)
		batch = np.full((batchsize, max_length), -1.0, dtype=np.float32)
		for i, data in enumerate(batch_array):
			batch[i,:len(data)] = data
		lm.reset_state()
		sum_loss += lm.learn(batch)
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	print "epoch:", epoch, "loss:", sum_loss / float(n_train), "time:", int(elapsed_time / 60), "min", "total_time:", int(total_time / 60), "min"
	lm.save(model_dir)


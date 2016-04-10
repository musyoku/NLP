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
n_train = 5000
batchsize = 64
total_time = 0
max_length_of_data = 100
length_limit = 15

for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		batch_array = []
		max_length_in_batch = 0
		for b in xrange(batchsize):
			length = length_limit + 1
			while length > length_limit:
				k = np.random.randint(0, n_dataset)
				data = dataset[k]
				length = len(data)
			batch_array.append(data)
			if length > max_length_in_batch:
				max_length_in_batch = length
		batch = np.full((batchsize, max_length_in_batch), -1.0, dtype=np.float32)
		for i, data in enumerate(batch_array):
			batch[i,:len(data)] = data
		lm.reset_state()
		sum_loss += lm.learn(batch)
		if t % 10 == 0:
			sys.stdout.write("\rLearning in progress...(%d / %d)" % (t, n_train))
			sys.stdout.flush()
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	sys.stdout.write("\r")
	print "epoch:", epoch, "loss:", sum_loss / float(n_train), "time:", int(elapsed_time / 60), "min", "total_time:", int(total_time / 60), "min", "length_limit:", length_limit
	sys.stdout.flush()
	lm.save(model_dir)
	if epoch % 10 == 0 and epoch != 0:
		length_limit = (length_limit + 5) if length_limit < max_length_of_data else max_length_of_data

# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import model
import vocab
from config import config

data_dir = "debug"
model_dir = "model"
dataset, config.n_vocab, config.n_dataset = vocab.load(data_dir)
reader = model.build()
reader.load(model_dir)

n_epoch = 1000
n_train = config.n_dataset * 10
total_time = 0

for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		k = np.random.randint(0, config.n_dataset)
		data = dataset[k]
		sum_loss += reader.train(data)
		if t % 10 == 0:
			sys.stdout.write("\rLearning in progress...(%d / %d)" % (t, n_train))
			sys.stdout.flush()
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	sys.stdout.write("\r")
	print "epoch:", epoch, "loss:", sum_loss / float(n_train), "time:", int(elapsed_time / 60), "min", "total_time:", int(total_time / 60), "min"
	sys.stdout.flush()
	reader.save(model_dir)

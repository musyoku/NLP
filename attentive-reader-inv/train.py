# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import model
import vocab
from config import config

data_dir = "alice"
model_dir = "model"
dataset, config.n_vocab, config.n_dataset = vocab.load(data_dir)
reader = model.build()
reader.load(model_dir)

n_epoch = 100000
n_train = 500
total_time = 0

def sample_data(limit=30):
	length = limit + 1
	while length > limit:
		k = np.random.randint(0, config.n_dataset)
		data = dataset[k]
		length = len(data)
	return data

for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		data = sample_data()
		sum_loss += reader.train(data)
		sys.stdout.write("\rLearning in progress...(%d / %d)" % (t, n_train))
		sys.stdout.flush()
	elapsed_time = time.time() - start_time
	total_time += elapsed_time
	sys.stdout.write("\r")
	print "epoch:", epoch, "loss:", sum_loss / float(n_train), "time:", int(elapsed_time / 60), "min", "total_time:", int(total_time / 60), "min"
	sys.stdout.flush()
	reader.save(model_dir)

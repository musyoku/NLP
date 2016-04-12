# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import model
import vocab
from config import config

data_dir = "text"
model_dir = "model"
dataset, config.n_vocab, config.n_dataset = vocab.load(data_dir)
lm = model.build()
lm.load(model_dir)

n_epoch = 1000
n_train = 5000
batchsize = 32
total_time = 0

# 長すぎるデータはメモリに乗らないこともあります
max_length_of_chars = 100

# 学習初期は短い文章のみ学習し、徐々に長くしていきます。
# この機能が必要ない場合は最初から大きな値を設定します。
current_length_limit = 15

for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		batch_array = []
		max_length_in_batch = 0
		for b in xrange(batchsize):
			length = current_length_limit + 1
			while length > current_length_limit:
				k = np.random.randint(0, config.n_dataset)
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
	print "epoch:", epoch, "loss:", sum_loss / float(n_train), "time:", int(elapsed_time / 60), "min", "total_time:", int(total_time / 60), "min", "current_length_limit:", current_length_limit
	sys.stdout.flush()
	lm.save(model_dir)
	if epoch % 10 == 0 and epoch != 0:
		current_length_limit = (current_length_limit + 5) if current_length_limit < max_length_of_chars else max_length_of_chars

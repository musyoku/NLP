# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from env import dataset, n_vocab, n_dataset, model, conf
from lattice import BigramLattice

n_epoch = 1
n_train = 1
batchsize = 1
total_time = 0

# 長すぎるデータはメモリに乗らないこともあります
max_length_of_chars = 100

# 学習初期は短い文章のみ学習し、徐々に長くしていきます。
# この機能が必要ない場合は最初から大きな値を設定します。
current_length_limit = 150
increasing_limit_interval = 1000

def sample_data():
	k = np.random.randint(0, n_dataset)
	k = 13
	data = dataset[k]
	return data

for epoch in xrange(n_epoch):
	start_time = time.time()
	sum_loss = 0
	for t in xrange(n_train):
		
		sentence = sample_data()
		encode = model.encode_word(sentence)
		print encode.data
		model.decode_word(encode.data[0])
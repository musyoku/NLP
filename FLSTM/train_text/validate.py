# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from args import args
from env import dataset, n_vocab, n_dataset, lstm, conf

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

# 学習時に長さ制限した場合は同じ値をここにもセット
current_length_limit = 150

def make_batch(batchsize):
	batch_array = []
	max_length_in_batch = 0
	for b in xrange(batchsize):
		length = current_length_limit + 1
		while length > current_length_limit:
			k = np.random.randint(0, n_dataset)
			data = dataset[k]
			length = len(data)
		batch_array.append(data)
		if length > max_length_in_batch:
			max_length_in_batch = length
	batch = np.full((batchsize, max_length_in_batch), -1.0, dtype=np.int32)
	for i, data in enumerate(batch_array):
		batch[i,:len(data)] = data
	return batch

def get_validation_data():
	max_length_in_batch = 0
	length = current_length_limit + 1
	while length > current_length_limit:
		k = np.random.randint(0, n_dataset)
		data = dataset[k]
		length = len(data)
	return data


# Validation
validation_batchsize = 1
lstm.reset_state()
phrase = make_batch(validation_batchsize)
result, forgets = lstm.predict_all(phrase, test=True, argmax=True)
for b in xrange(validation_batchsize):
	print "batch", b
	str = ""
	for j in xrange(len(result)):
		cc = phrase[b, j]
		if cc < 1:
			break
		str += vocab.id_to_word(cc)
	print str
		
	str = ""
	for j in xrange(len(result)):
		cp = result[j][b]
		f = forgets[j].data
		cc = phrase[b, j]
		if cc < 1:
			break
		print f
		str += vocab.id_to_word(cp)
	print str

# for phrase in xrange(100):
# 	lstm.reset_state()
# 	str = ""
# 	char = get_validation_data()[0]
# 	if char == 0:
# 		continue
# 	for n in xrange(100):
# 		str += vocab.id_to_word(char)
# 		id = lstm.predict(char, test=True, argmax=False)
# 		if id == 0:
# 			break
# 		char = id
# 		print lstm.prev_forget.data
# 	print str
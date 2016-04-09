# -*- coding: utf-8 -*-
import os
import numpy as np

vocab = {}
inv_vocab = {}

def load_data(dir):
	fs = os.listdir(dir)
	print "loading", len(fs), "files..."
	dataset = None
	for fn in fs:
		words = open("%s/%s" % (dir, fn)).read().replace("\n", "<eos>").strip().split()
		data = np.empty((len(words),), dtype=np.int32)
		for i, word in enumerate(words):
			if word not in vocab:
				vocab[word] = len(vocab)
			data[i] = vocab[word]
		if dataset is None:
			dataset = data
		else:
			dataset = np.append(dataset, data)
	return dataset
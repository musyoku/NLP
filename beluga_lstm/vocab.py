# -*- coding: utf-8 -*-
import numpy as np

vocab = {}
inv_vocab = {}

def load_data(dir):
	words = open(filename).read().replace('\n', '<eos>').strip().split()
	dataset = np.ndarray((len(words),), dtype=np.int32)
	for i, word in enumerate(words):
		if word not in vocab:
			vocab[word] = len(vocab)
		dataset[i] = vocab[word]
	return dataset
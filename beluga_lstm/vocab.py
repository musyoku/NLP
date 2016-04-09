# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs

vocab = {}
inv_vocab = {}

def load(dir):
	fs = os.listdir(dir)
	print "loading", len(fs), "files..."
	dataset = []
	for fn in fs:
		unko = codecs.open("%s/%s" % (dir, fn), "r", "utf_8_sig")
		for line in unko:
			data = np.empty((len(line),), dtype=np.int32)
			for i in xrange(len(line)):
				word = line[i]
				if word not in vocab:
					vocab[word] = len(vocab)
				data[i] = vocab[word]
			dataset.append(data)
	print "# of chars:", len(vocab)
	print "# of data:", len(dataset)
	return dataset
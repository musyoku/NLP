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
	vocab["<unk>"] = -1
	vocab["<bos>"] = 0
	vocab["<eos>"] = 1
	inv_vocab[vocab["<bos>"]] = "<bos>"
	inv_vocab[vocab["<eos>"]] = "<eos>"
	for fn in fs:
		lines = codecs.open("%s/%s" % (dir, fn), "r", "utf_8_sig")	# BOMありならutf_8_sig　そうでないならutf_8
		for line in lines:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			data = np.empty((len(line)+2,), dtype=np.int32)
			data[0] = vocab["<bos>"]
			for i in xrange(len(line)):
				word = line[i]
				if word not in vocab:
					vocab[word] = len(vocab)
					inv_vocab[vocab[word]] = word
				data[i+1] = vocab[word]
				data[-1] = vocab["<eos>"]
			dataset.append(data)
	n_vocab = len(vocab)
	n_dataset = len(dataset)
	print "# of chars:", n_vocab
	print "# of data:", n_dataset
	return dataset, n_vocab, n_dataset

def id_to_word(id):
	if id < 0:
		return "<unk>"
	return inv_vocab[id]

def ids_to_str(ids):
	str = ""
	for i, id in enumerate(ids):
		if id == 0:
			break
		str += id_to_word(id)
	return str

def word_to_id(word):
	return vocab[word]
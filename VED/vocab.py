# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs

vocab = {}
inv_vocab = {}

eos_id = 0
bos_id = 1

def load(dir):
	fs = os.listdir(dir)
	print "loading", len(fs), "files..."
	dataset = []
	vocab["<eos>"] = eos_id
	inv_vocab[eos_id] = "<eos>"
	vocab["<bos>"] = bos_id
	inv_vocab[bos_id] = "<bos>"
	for fn in fs:
		unko = codecs.open("%s/%s" % (dir, fn), "r", "utf_8_sig")	# BOMありならutf_8_sig　そうでないならutf_8
		for line in unko:
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			if len(line) > 0:
				# data = np.empty((len(line) + 1,), dtype=np.int32)
				data = np.empty((len(line) + 2,), dtype=np.int32)
				data[0] = bos_id
				for i in xrange(len(line)):
					word = line[i]
					if word not in vocab:
						vocab[word] = len(vocab)
						inv_vocab[vocab[word]] = word
					# data[i] = vocab[word]
					data[i + 1] = vocab[word]
				# data[len(line)] = eos_id
				data[len(line) + 1] = eos_id
				dataset.append(data)
	n_vocab = len(vocab)
	n_dataset = len(dataset)
	print "# of chars:", n_vocab
	print "# of data:", n_dataset
	return dataset, n_vocab, n_dataset

def id_to_word(id):
	return inv_vocab[id]

def ids_to_str(ids):
	str = ""
	for i in xrange(len(ids)):
		str += id_to_word(ids[i])
	return str

def word_to_id(word):
	return vocab[word]
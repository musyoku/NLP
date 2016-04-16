# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
import model
import vocab
from config import config

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

data_dir = "text"
model_dir = "model"
dataset, config.n_vocab, config.n_dataset = vocab.load(data_dir)
lm = model.build()
lm.load(model_dir)

def sample_seq():
	target_batch_array = []
	max_length_in_batch = 0
	k = np.random.randint(0, config.n_dataset)
	target_seq = dataset[k]
	source_seq = target_seq[::-1]
	return source_seq, target_seq

for phrase in xrange(50):
	source_seq, target_seq = sample_seq()
	lm.reset_state()
	y_seq = lm.decode(source_seq, sampling_y=True, test=True)
	print "source_seq", source_seq
	print vocab.ids_to_str(source_seq[::-1])
	print "target_seq", target_seq
	print "decode", y_seq
	print vocab.ids_to_str(y_seq)

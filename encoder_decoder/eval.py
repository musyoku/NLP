# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
import model
import vocab
from config import config

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

data_dir = "debug"
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
	lm.reset_state()
	source_seq, target_seq = sample_seq()
	y_ids = lm.decode(source_seq)
	print "source_seq", source_seq
	print "target_seq", target_seq
	print "decode", y_ids
	print vocab.ids_to_str(y_ids)

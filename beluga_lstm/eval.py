# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
import model
import vocab

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

data_dir = "text"
model_dir = "model"
dataset, n_vocab, n_dataset = vocab.load(data_dir)
lm = model.build(n_vocab)
lm.load(model_dir)

for phrase in xrange(50):
	lm.reset_state()
	str = ""
	char = dataset[np.random.randint(0, n_dataset)][0]
	for n in xrange(1000):
		dist = lm.distribution(char)[0]
		id = np.random.choice(np.arange(n_vocab, dtype=np.uint8), 1, p=dist)[0]
		if id == 0:
			break
		str += vocab.id_to_word(char)
		char = id
	print str


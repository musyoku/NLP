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

for phrase in xrange(50):
	lm.reset_state()
	str = ""
	char = dataset[np.random.randint(0, config.n_dataset)][0]
	for n in xrange(1000):
		str += vocab.id_to_word(char)
		dist = lm.distribution(char, test=True)[0]
		id = np.random.choice(np.arange(config.n_vocab, dtype=np.uint8), 1, p=dist)[0]
		if id == 0:
			break
		char = id
	print str


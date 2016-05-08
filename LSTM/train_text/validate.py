# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from args import args
from env import dataset, n_vocab, n_dataset, lstm, conf

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

for phrase in xrange(50):
	lstm.reset_state()
	str = ""
	char = dataset[np.random.randint(0, n_dataset)][0]
	for n in xrange(100):
		str += vocab.id_to_word(char)
		id = lstm.predict(char, test=True)
		if id == 0:
			break
		char = id
	print str


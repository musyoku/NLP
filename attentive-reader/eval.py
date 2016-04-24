# -*- coding: utf-8 -*-
import os, sys, time, codecs, pylab
import numpy as np
import pandas as pd
import seaborn as sns
import model
import vocab
from config import config

sns.set()

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

data_dir = "debug"
model_dir = "model"
dataset, config.n_vocab, config.n_dataset = vocab.load(data_dir)
reader = model.build()
reader.load(model_dir)

def sample_data():
	k = np.random.randint(0, config.n_dataset)
	data = dataset[k]
	return data

for repeat in xrange(1):
	data = sample_data()
	df_attention = pd.DataFrame(columns=["t", "character", "weight"])
	for pos in xrange(len(data)):
		weight, predicted_char_embed = reader.forward_one_step(data, pos, test=True)
		onehot = reader.inverse_embed(predicted_char_embed.data[0])
		predicted_char = np.argmax(onehot)
		for t in xrange(len(data)):
			df_attention = df_attention.append([{"t": pos, "character": data[t], "weight": weight[0, t]}], ignore_index=True)
	df_attention_pivot = pd.pivot_table(data=df_attention, values="weight", columns="t", index="character", aggfunc=np.mean)
	pylab.figure(figsize=(12, 9))
	sns.heatmap(df_attention_pivot, annot=False, fmt="g", cmap="Blues")
	pylab.yticks(rotation=0) 
	pylab.show()

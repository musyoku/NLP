# -*- coding: utf-8 -*-
import os, sys, time, codecs, pylab
import numpy as np
import pandas as pd
import seaborn as sns
import model
import vocab
from config import config
import matplotlib.font_manager
fontprop = matplotlib.font_manager.FontProperties(fname="NotoSansJP-Medium.otf")

sns.set()

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

data_dir = "alice"
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
	df_attention = pd.DataFrame(columns=["t", "character", "weight", "phrase"])
	length = len(data)
	mask = np.zeros((length, length), dtype=np.bool)
	for pos in xrange(length):
		weight, predicted_char_embed = reader.forward_one_step(data, pos, test=True)
		onehot = reader.inverse_embed(predicted_char_embed.data[0])
		predicted_char = np.argmax(onehot)
		for t in xrange(length):
			df_attention = df_attention.append([{"t": pos, "character": vocab.id_to_word(data[t]), "weight": weight[0, t], "phrase": t}], ignore_index=True)
			if t == pos:
				mask[pos, length - t - 1] = True
	df_attention_pivot = pd.pivot_table(data=df_attention, values="weight", columns="t", index="phrase", aggfunc=np.mean)
	pylab.figure(figsize=(16, 12))
	ax = sns.heatmap(df_attention_pivot, annot=False, fmt="d", cmap="Blues", mask=mask, square=True)
	ax.set_yticklabels(df_attention[df_attention.t == 0]["character"], rotation=90, va="center", size="x-large")
	ax.set_xticklabels(np.arange(length, dtype=np.int32))
	pylab.show()

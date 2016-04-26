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

data_dir = "beluga"
model_dir = "model"
result_dir = "result"
dataset, config.n_vocab, config.n_dataset = vocab.load(data_dir)
reader = model.OneDirectionAttentiveReader()
reader.load(model_dir)

try:
	os.mkdir(result_dir)
except:
	pass

def sample_data(limit=30):
	length = limit + 1
	while length > limit:
		k = np.random.randint(0, config.n_dataset)
		data = dataset[k]
		length = len(data)
	return data

for repeat in xrange(20):
	data = sample_data()
	df_attention = pd.DataFrame(columns=["t", "character", "weight", "phrase"])
	length = len(data)
	mask = np.zeros((length, length), dtype=np.bool)
	num_correct = 0
	for pos in xrange(length):
		weight, predicted_char_embed = reader.forward_one_step(data, pos, test=True)
		onehot = reader.inverse_embed(predicted_char_embed.data[0])
		predicted_char = np.argmax(onehot)
		if predicted_char == data[pos]:
			num_correct += 1
		for t in xrange(length):
			df_attention = df_attention.append([{"t": pos, "character": vocab.id_to_word(data[t]), "weight": weight[0, t], "phrase": t}], ignore_index=True)
			if t == pos:
				mask[length - pos - 1, length - t - 1] = True
	print repeat, ":"
	print num_correct, "/", length
	df_attention_pivot = pd.pivot_table(data=df_attention, values="weight", columns="phrase", index="t", aggfunc=np.mean)
	with sns.axes_style("white"):
		pylab.figure(figsize=(16, 12))
		ax = sns.heatmap(df_attention_pivot, annot=False, fmt="d", cmap="Blues", mask=mask, square=True, vmin=0.0, vmax=1.0)
		ax.set_xticklabels(df_attention[df_attention.t == 0]["character"], size="x-large", fontproperties=fontprop)
		ax.set_yticklabels(np.arange(length, dtype=np.int32)[::-1], va="center")
		ax.xaxis.tick_top()
		filename = "%s/%d.png" % (result_dir, repeat)
		pylab.savefig(filename)
		print filename, "saved."

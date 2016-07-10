# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from args import args
from env import dataset, n_vocab, n_dataset, model, conf
from lattice import BigramLattice

sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

n_epoch = 1000
n_train = 100
batchsize = 64
total_time = 0

current_length_limit = 50

def sample_batch():
	batch_array = []
	max_length_in_batch = 0
	for b in xrange(batchsize):
		length = current_length_limit + 1
		while length > current_length_limit:
			k = np.random.randint(0, n_dataset)
			data = dataset[k]
			length = len(data)
		batch_array.append(data)
		if length > max_length_in_batch:
			max_length_in_batch = length
	batch = np.full((batchsize, max_length_in_batch), -1, dtype=np.int32)
	for i, data in enumerate(batch_array):
		batch[i,:len(data)] = data
	return batch

def sample_data():
	length = current_length_limit + 1
	while length > current_length_limit:
		k = np.random.randint(0, n_dataset)
		data = dataset[k]
		length = len(data)
	return data

def train_single():
	for epoch in xrange(n_epoch):
		start_time = time.time()
		sum_loss = 0
		for t in xrange(n_train):
			
			sentence = sample_data()
			loss_reconstruction, loss_generator, loss_discriminator = model.train_word_embedding(sentence)
			print loss_reconstruction, loss_generator, loss_discriminator

		model.save(args.model_dir)

		for i in xrange(10):
			sentence = sample_data()
			encode = model.encode_word(sentence, test=True)
			char_ids = model.decode_word(encode.data[0], test=True)
			print "target: ", vocab.ids_to_str(sentence)
			print "predict:", vocab.ids_to_str(char_ids)
			print ""

def train_batch():
	for epoch in xrange(n_epoch):
		start_time = time.time()
		sum_loss = 0
		for t in xrange(n_train):
			
			sentence_batch = sample_batch()
			loss_reconstruction, loss_generator, loss_discriminator = model.train_word_embedding_batch(sentence_batch)
			print loss_reconstruction, loss_generator, loss_discriminator

		model.save(args.model_dir)

		for i in xrange(50):
			sentence = sample_data()
			encode = model.encode_word(sentence, test=True)
			char_ids = model.decode_word(encode.data[0], argmax=True, test=True)
			print "target: ", vocab.ids_to_str(sentence)
			print "predict:", vocab.ids_to_str(char_ids)
			print ""

lattice = BigramLattice(model, vocab)
sentence = sample_data()
segmentation = lattice.segment(sentence)
print segmentation

lattice = BigramLattice(model, vocab)
sentence = sample_data()
segmentation = lattice.segment(sentence)
print segmentation

# train_batch()
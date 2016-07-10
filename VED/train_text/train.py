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

current_length_limit = 30

def sample_initial_word():
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
	batch = np.full((batchsize, max_length_in_batch - 1), -1, dtype=np.int32)
	for i, data in enumerate(batch_array):
		batch[i,:len(data)-1] = data[1:]
	return batch

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
	return data, k

def train_single():
	for epoch in xrange(n_epoch):
		start_time = time.time()
		sum_loss = 0
		for t in xrange(n_train):
			
			sentence, _ = sample_data()
			loss_reconstruction, loss_generator, loss_discriminator = model.train_word_embedding(sentence)
			print loss_reconstruction, loss_generator, loss_discriminator

		model.save(args.model_dir)

		for i in xrange(10):
			sentence, _ = sample_data()
			encode = model.encode_word(sentence, test=True)
			char_ids = model.decode_word(encode.data[0], test=True)
			print "target: ", vocab.ids_to_str(sentence)
			print "predict:", vocab.ids_to_str(char_ids)
			print ""

def train_batch():
	n_epoch = 10
	n_train = 100
	for epoch in xrange(n_epoch):
		start_time = time.time()
		sum_loss = 0
		for t in xrange(n_train):
			
			sentence_batch = sample_initial_word()
			loss_reconstruction, loss_generator, loss_discriminator = model.train_word_embedding_batch(sentence_batch)
			print loss_reconstruction, loss_generator, loss_discriminator

			sentence_batch = np.asanyarray([[0], [1]], dtype=np.int32)
			loss_reconstruction, loss_generator, loss_discriminator = model.train_word_embedding_batch(sentence_batch)

		model.save(args.model_dir)

		for i in xrange(50):
			sentence, _ = sample_data()
			encode = model.encode_word(sentence, test=True)
			char_ids = model.decode_word(encode.data[0], argmax=True, test=True)
			print "target: ", vocab.ids_to_str(sentence)
			print "predict:", vocab.ids_to_str(char_ids)
			print ""

def train_initial_ngram():
	n_epoch = 10
	n_train = 100
	batchsize
	for epoch in xrange(n_epoch):
		for t in xrange(n_train):
			sentence_batch = sample_initial_word()
			words = []
			words.append(np.full((batchsize, 1), 1, dtype=np.int32))
			words.append(sentence_batch)
			words.append(np.full((batchsize, 1), 0, dtype=np.int32))
			loss = model.train_word_ngram_sequence_batch(words[:-1], words[1:])
			print loss
		model.save(args.model_dir)

def train_word_ngram():
	lattice = BigramLattice(model, vocab)

	batchsize = 1
	# indices = [1323, 3194, 2801, 3132, 2883, 3930, 1280, 1922, 143, 528, 1599, 2974, 613, 3558, 1218, 3462]
	for epoch in xrange(n_epoch):
		start_time = time.time()
		for t in xrange(n_train):
			sentence, k = sample_data()
			print "data", k
			print vocab.ids_to_str(sentence)
			segmentation = lattice.segment(sentence)
			words = []
			max_length = 0
			start = 0
			for s in xrange(len(segmentation)):
				word = sentence[start:start + segmentation[s]]
				print vocab.ids_to_str(word), " / ",
				if word[0] == 0 or word[0] == 1:
					pass
				else:
					word = np.r_[word, np.asarray([0], dtype=np.int32)]
				words.append(word)
				if len(word) > max_length:
					max_length = len(word)
				start += segmentation[s]
			print

			char_ids_batch = np.full((len(words), max_length), -1, dtype=np.int32)
			for i in xrange(len(words)):
				word = words[i]
				char_ids_batch[i, :len(word)] = word

			# word embedding
			n_steps = 50
			sum_loss_r = 0
			sum_loss_g = 0
			sum_loss_d = 0
			for step in xrange(n_steps):
				loss_r, loss_g, loss_d = model.train_word_embedding_batch(char_ids_batch)
				sum_loss_r += loss_r
				sum_loss_g += loss_g
				sum_loss_d += loss_d
			print sum_loss_r / n_steps, sum_loss_g / n_steps, sum_loss_d / n_steps

			# word n-gram
			sum_loss = 0
			for step in xrange(n_steps):
				loss = model.train_word_ngram_sequence(words[:-1], words[1:])
				sum_loss += loss
			print sum_loss / n_steps

		elapsed_time = time.time() - start_time
		print "	time", elapsed_time
		model.save(args.model_dir)

# train_batch()
# train_initial_ngram()
train_word_ngram()
# train_batch()
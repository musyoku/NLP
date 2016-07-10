import math
import numpy as np

class Lattice:
	def __init__(self, model, vocab):
		self.pre_computation = False
		self.reserved_words = []
		self.reserved_contexts = []
		self.index = 0
		self.pre_computed_pw_h = None
		self.model = model
		self.vocab = vocab

class BigramLattice(Lattice):
	def compute_alpha_t_k(self, sentence_ids, alpha, t, k):
		if t < 1 or k < 1:
			return
		word = sentence_ids[t - k + 1:t + 1]
		if t - k == 0:
			context = np.asarray([sentence_ids[0]], dtype=np.int32)
			if self.pre_computation:
				self.reserved_words.append(word)
				self.reserved_contexts.append(context)
			else:
				alpha[t, k] = self.pre_computed_pw_h[self.index]
				self.index += 1
			return

		_sum = 0
		for j in xrange(1, t - k + 1):
			context = sentence_ids[t - k - j + 1:t - k + 1]
			if self.pre_computation:
				self.reserved_words.append(word)
				self.reserved_contexts.append(context)
				pw_h = 0
			else:
				pw_h =  self.pre_computed_pw_h[self.index]
				self.index += 1
			_sum += pw_h * alpha[t - k, j]
		alpha[t, k] = _sum

	def forward_filtering(self, sentence_ids, alpha):
		for t in xrange(1, len(sentence_ids) - 1):
			for k in xrange(1, t + 1):
				self.compute_alpha_t_k(sentence_ids, alpha, t, k)

	def backward_sampling(self, sentence_ids, alpha, segmentation):
		k = self.sapmle_starting_k(sentence_ids, alpha)
		segmentation.append(k)
		t = len(sentence_ids) - 2 - k
		while t > 0:
			k = self.sample_backward_k(sentence_ids, alpha, t)
			segmentation.append(k)
			t -= k

	def sapmle_starting_k(self, sentence_ids, alpha):
		p_k = []
		sum_p = 0
		eos = np.asarray([self.vocab.eos_id], dtype=np.int32)
		t = len(sentence_ids) - 2
		for k in xrange(1, len(sentence_ids) - 1):
			context = sentence_ids[- k - 1:- 1]
			p = self.model.Pw_h(eos, context) * alpha[t, k]
			sum_p += p
			p_k.append(p)

		r = np.random.uniform(0, sum_p)
		sum_p = 0
		for k_i in xrange(len(p_k)):
			sum_p += p_k[k_i]
			if r < sum_p:
				return k_i + 1

		return len(p_k)

	def sample_backward_k(self, sentence_ids, alpha, t):
		if t == 1:
			return 1
		p_k = []
		sum_p = 0
		for k in xrange(1, t + 1):
			p = alpha[t, k]
			sum_p += p
			p_k.append(p)

		r = np.random.uniform(0, sum_p)
		sum_p = 0
		for k_i in xrange(len(p_k)):
			sum_p += p_k[k_i]
			if r < sum_p:
				return k_i + 1

		return len(p_k)

	def segment(self, sentence_ids):
		print sentence_ids
		alpha = np.zeros((len(sentence_ids) + 1, len(sentence_ids) + 1), dtype=np.float64)
		alpha[0, 1] = 0
		segmentation = []

		self.pre_computation = True
		self.forward_filtering(sentence_ids, alpha)

		max_word_length = len(sentence_ids) - 2
		max_context_length = max_word_length - 1
		word_char_ids_batch = np.full((len(self.reserved_words), max_word_length), -1, dtype=np.float32)
		for i in xrange(len(self.reserved_words)):
			char_ids = self.reserved_words[i]
			word_char_ids_batch[i, :len(char_ids)] = char_ids

		context_char_ids_batch = np.full((len(self.reserved_contexts), max_context_length), -1, dtype=np.float32)
		for i in xrange(len(self.reserved_contexts)):
			char_ids = self.reserved_contexts[i]
			context_char_ids_batch[i, :len(char_ids)] = char_ids

		n_rows_per_split = 500
		division = int(math.ceil(len(self.reserved_words) / float(n_rows_per_split)))

		word_char_ids_batch_array = []
		context_char_ids_batch_array = []

		if division == 0:
			word_char_ids_batch_array.append(word_char_ids_batch)
			context_char_ids_batch_array.append(context_char_ids_batch)
		else:
			for i in xrange(division - 1):
				word_char_ids_batch_array.append(word_char_ids_batch[i * n_rows_per_split:(i + 1) * n_rows_per_split])
				context_char_ids_batch_array.append(context_char_ids_batch[i * n_rows_per_split:(i + 1) * n_rows_per_split])
			i = division - 1
			word_char_ids_batch_array.append(word_char_ids_batch[i * n_rows_per_split:])
			context_char_ids_batch_array.append(context_char_ids_batch[i * n_rows_per_split:])

		for i in xrange(division):
			pw_h_batch = self.model.Pw_h_batch(word_char_ids_batch_array[i], context_char_ids_batch_array[i])
			if self.pre_computed_pw_h is None:
				self.pre_computed_pw_h = pw_h_batch
			else:
				self.pre_computed_pw_h = np.r_[self.pre_computed_pw_h, pw_h_batch]
		self.index = 0

		self.pre_computation = False
		segmentation.append(1)
		self.forward_filtering(sentence_ids, alpha)
		self.backward_sampling(sentence_ids, alpha, segmentation)
		segmentation.append(1)

		return segmentation

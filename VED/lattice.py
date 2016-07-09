class Lattice:
	def __init__(self, model, vocab):
		self.model = model
		self.vocab = vocab

class BigramLattice(Lattice):
	def compute_alpha_t_k(self, sentence_ids, alpha, t, k):
		if t < 1 or k < 1:
			return

	def forward_filtering(self):
		pass

	def backward_sampling(self):
		pass

	def sapmle_starting_k(self):
		pass

	def sample_backward_k(self):
		pass

	def segment(self):
		pass

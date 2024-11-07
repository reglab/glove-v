import numpy as np
from gensim.models import KeyedVectors

from .data import get_variances


class GloVeVEmbeddings(KeyedVectors):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variances = get_variances()  # Load your variances here

    def get_vector_with_variance(self, word):
        return self[word], self.variances.get(word, np.zeros((300, 300)))


def load_variance_embeddings(path):
    # Load and return GloVeVEmbeddings instance
    pass

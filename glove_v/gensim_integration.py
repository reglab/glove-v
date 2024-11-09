import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

import glove_v.utils.file as file_utils

from .propagate import sample_vector
from .variance import load_variance
from .vector import load_vectors, load_vocab


class GloVeVKeyedVectors:
    """
    A class for working with GloVe-V embeddings using Gensim.
    """

    def __init__(
        self,
        embedding_name: str,
        download_dir: str = f"{file_utils.get_data_path()}/glove-v",
    ):
        vocab, _ = load_vocab(embedding_name=embedding_name)

        self.embedding_name = embedding_name
        self.V = len(vocab)
        self.vocab = vocab
        self.vectors = load_vectors(embedding_name, download_dir, "gensim")
        self.d = self.vectors.vector_size

    def sample_vectors(
        self,
        approximation: bool,
        sample_size: int = 1,
        verbose: bool = False,
        store: bool = False,
    ):
        """
        Sample vectors from the GloVe-V embedding model. If store is True, the variances are stored in memory to avoid reloading
        at every single sample.

        Args:
            approximation (bool): Whether to use the approximate variance or the exact variance.
            sample_size (int): The number of samples to draw.
            verbose (bool): Whether to print progress.
            store (bool): Whether to store the variances in memory.
        """

        variances = None
        if store:
            variances = {}
            for word_idx in tqdm(
                self.vocab.values(), desc="Loading variances", disable=not verbose
            ):
                word_var = load_variance(
                    embedding_name=self.embedding_name,
                    approximation=approximation,
                    word_idx=word_idx,
                )
                variances[word_idx] = word_var

        sampled_matrix = np.zeros((sample_size, self.V, self.d))
        for word, word_idx in self.vocab.items():
            if store:
                word_var = variances[word_idx]
            else:
                word_var = load_variance(
                    embedding_name=self.embedding_name,
                    approximation=approximation,
                    word_idx=word_idx,
                )
            sample_matrix_word = sample_vector(
                variance=word_var,
                vector=self.vectors[word],
                n=sample_size,
            )
            sampled_matrix[:, word_idx, :] = sample_matrix_word

        sampled_vectors = []
        for k in range(sample_size):
            kv = KeyedVectors(self.d)
            kv.vectors = sampled_matrix[k]
            kv.index_to_key = list(self.vocab.keys())
            kv.key_to_index = self.vocab
            sampled_vectors.append(kv)

        return sampled_vectors

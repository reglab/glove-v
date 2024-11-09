from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from safetensors import safe_open

from glove_v.utils import file as file_utils


def load_vocab(
    embedding_name: str, download_dir: str = f"{file_utils.get_data_path()}/glove-v"
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Loads dictionaries of word-to-index and index-to-word vocabulary conversions.

    Args:
        embedding_name: (str) Name of the embedding to load
        download_dir: (str) Path to the directory where the embedding is saved
    """
    try:
        vocab_path = Path(download_dir) / embedding_name / "vocab.txt"
        with vocab_path.open() as f:
            words = [x.rstrip().split(" ")[0] for x in f.readlines()]
    except FileNotFoundError as err:
        raise file_utils.file_loading_error_message(
            "vocab.txt", download_dir, embedding_name
        ) from err

    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = dict(enumerate(words))
    return vocab, ivocab


def load_vectors(
    embedding_name: str,
    download_dir: str = f"{file_utils.get_data_path()}/glove-v",
    format: str = "numpy",
) -> dict[str, np.ndarray] | np.ndarray | KeyedVectors:
    possible_formats = ["numpy", "gensim", "dictionary"]
    assert (
        format in possible_formats
    ), f"Format should be one of the following: {possible_formats}"

    try:
        with safe_open(
            f"{download_dir}/{embedding_name}/vectors.safetensors", framework="numpy"
        ) as f:
            vectors = f.get_tensor("center_vectors")
    except FileNotFoundError as err:
        raise file_utils.file_loading_error_message(
            "vectors.safetensors", download_dir, embedding_name
        ) from err

    vocab, ivocab = load_vocab(embedding_name, download_dir)
    if format == "numpy":
        return vectors
    elif format == "dictionary":
        return {ivocab[idx]: vectors[idx] for idx in range(vectors.shape[0])}
    elif format == "gensim":
        kv = KeyedVectors(vectors.shape[1])
        kv.vectors = vectors
        kv.index_to_key = list(vocab.keys())
        kv.key_to_index = vocab
        return kv

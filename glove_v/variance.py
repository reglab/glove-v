import functools
from pathlib import Path

import numpy as np
from safetensors import safe_open

from glove_v import vector
from glove_v.utils import file as file_utils


@functools.lru_cache(maxsize=10)
def load_approx_info(
    embedding_name: str, download_dir: str = f"{file_utils.get_data_path()}/glove-v"
) -> dict[int, tuple[str, int | None]]:
    # Load vocabulary
    vocab_path = Path(download_dir) / embedding_name / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            file_utils.file_loading_error_message(
                "vocab.txt", download_dir, embedding_name
            )
        )
    vocab, _ = vector.load_vocab(
        embedding_name=embedding_name,
    )

    approx_info = {}
    approx_info_path = Path(download_dir) / embedding_name / "approx_info.txt"
    if not approx_info_path.exists():
        raise FileNotFoundError(
            file_utils.file_loading_error_message(
                "approx_info.txt", download_dir, embedding_name
            )
        )
    with approx_info_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            approx_type = parts[1]
            rank = parts[2] if len(parts) > 2 else None
            approx_info[vocab[word]] = (approx_type, rank)
    return approx_info


@functools.lru_cache(maxsize=10)
def load_chunk_map(
    embedding_name: str, download_dir: str = f"{file_utils.get_data_path()}/glove-v"
) -> dict[int, tuple[int, int]]:
    chunk_map_path = Path(download_dir) / embedding_name / "chunk_map.txt"
    if not chunk_map_path.exists():
        raise FileNotFoundError(
            file_utils.file_loading_error_message(
                "chunk_map.txt", download_dir, embedding_name
            )
        )

    # Each row contains: word    global_word_index    chunk_index    local_chunk_index
    chunk_map = {}
    with Path(chunk_map_path).open("r") as f:
        for line in f:
            word, global_idx, chunk_idx, local_idx = line.strip().split("\t")
            chunk_idx = int(chunk_idx)
            chunk_map[int(global_idx)] = (chunk_idx, int(local_idx))
    return chunk_map


def load_variance(
    embedding_name: str,
    word_idx: int,
    approximation: bool = True,
    download_dir: str = f"{file_utils.get_data_path()}/glove-v",
) -> np.ndarray:
    """
    Reconstruct the approximated variance matrix for word at index i from safetensor file.

    Args:
        embedding_name: Name of the embedding to load
        word_idx: Index of the word in the vocabulary
        download_dir: Path to the directory where the embedding is saved

    Returns:
        np.ndarray: Reconstructed GloVe-V variance matrix
    """
    chunk_map = load_chunk_map(embedding_name, download_dir)
    chunk_idx, local_idx = chunk_map[word_idx]

    approx_info = load_approx_info(embedding_name, download_dir)
    approx_type, rank = approx_info[word_idx]

    if approximation:
        approximation_path = (
            Path(download_dir) / embedding_name / f"approxchunk_{chunk_idx}.safetensors"
        )
        if not approximation_path.exists():
            raise FileNotFoundError(
                file_utils.file_loading_error_message(
                    f"approxchunk_{chunk_idx}.safetensors", download_dir, embedding_name
                )
            )

        with safe_open(approximation_path, framework="numpy") as f:
            if approx_type == "diagonal":
                diagonal = f.get_tensor(f"diag_{word_idx}")
                return np.diag(diagonal)

            # SVD approximation
            if approx_type == "svd":
                U = f.get_tensor(f"U_{word_idx}")
                s = f.get_tensor(f"s_{word_idx}")
                Vt = f.get_tensor(f"Vt_{word_idx}")

                # Reconstruct using SVD components: U * diag(s) * Vt
                return U @ np.diag(s) @ Vt

            # Complete approximation (SVD unavailable)
            elif approx_type == "complete":
                complete = f.get_tensor(f"complete_{word_idx}")
                return complete

            else:
                raise KeyError(
                    f"[ERROR No approximation found for word index {word_idx}"
                )

    else:
        chunk_path = (
            Path(download_dir)
            / embedding_name
            / f"completechunk_{chunk_idx}.safetensors"
        )
        if not chunk_path.exists():
            raise FileNotFoundError(
                file_utils.file_loading_error_message(
                    f"completechunk_{chunk_idx}.safetensors",
                    download_dir,
                    embedding_name,
                )
            )

        with safe_open(chunk_path, framework="numpy") as f:
            word_var = f.get_tensor("variances")[local_idx]
            return word_var

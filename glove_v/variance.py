from pathlib import Path

import numpy as np
from safetensors import safe_open

from glove_v.utils import file as file_utils


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
    if approximation:
        approximation_path = (
            Path(download_dir) / embedding_name / "ApproximationVariances.safetensors"
        )
        if not approximation_path.exists():
            raise FileNotFoundError(
                file_utils.file_loading_error_message(
                    "ApproximationVariances.safetensors", download_dir, embedding_name
                )
            )

        with safe_open(approximation_path, framework="numpy") as f:
            # Check if this is a diagonal approximation
            if f"diag_{word_idx}" in f.keys():
                diagonal = f.get_tensor(f"diag_{word_idx}")
                return np.diag(diagonal)

            # Otherwise, it must be an SVD approximation
            elif f"U_{word_idx}" in f.keys():
                U = f.get_tensor(f"U_{word_idx}")
                s = f.get_tensor(f"s_{word_idx}")
                Vt = f.get_tensor(f"Vt_{word_idx}")

                # Reconstruct using SVD components: U * diag(s) * Vt
                return U @ np.diag(s) @ Vt

            else:
                raise KeyError(
                    f"[ERROR No approximation found for word index {word_idx}"
                )

    else:
        complete_path = (
            Path(download_dir) / embedding_name / "CompleteVariances.safetensors"
        )
        if not complete_path.exists():
            raise FileNotFoundError(
                file_utils.file_loading_error_message(
                    "CompleteVariances.safetensors", download_dir, embedding_name
                )
            )
        with safe_open(complete_path, framework="numpy") as f:
            word_var = f.get_tensor("variances")[word_idx]
            return word_var

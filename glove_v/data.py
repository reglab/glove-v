"""
This module contains functions to download the GloVe embeddings and GloVe-V variances from HuggingFace. The
download can be either for the approximate variances or the original variances. The approximate variances are
more lightweight, and guarantee 90% reconstruction of the original variance for each word.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

import glove_v.utils.file as file_utils

AVAILABLE_EMBEDDINGS = [
    "Toy-Embeddings",
    "COHA_1900-1999_300d",
]


def download_embeddings(
    embedding_name: str,
    approximation: bool = True,
    download_dir: str = f"{file_utils.get_data_path()}/glove-v",
) -> None:
    """
    Downloads the vectors and variances for a selected corpus.

    Args:
        embedding_name: (str) The specific embedding to download. This should match one of the keys in the AVAILABLE_EMBEDDINGS dictionary
        approximation: (bool) Whether to download the approximate or complete GloVe-V variances. The GloVe embeddings
        are the same for both cases.
        download_dir: (str) path where GloVe-V files should be saved
    """
    if embedding_name not in AVAILABLE_EMBEDDINGS:
        raise ValueError(
            f"[ERROR] Embeddings should be one of the following: {AVAILABLE_EMBEDDINGS}"
        )

    final_download_dir = Path(download_dir) / embedding_name
    final_download_dir.mkdir(parents=True, exist_ok=True)

    # Download vocabulary, embeddings and support files
    for file in ["vocab.txt", "vectors.safetensors", "chunk_map.txt"]:
        file_path = final_download_dir / file
        if not file_path.exists():
            downloaded_path = hf_hub_download(
                repo_id="reglab/glove-v",
                filename=f"{embedding_name}/{file}",
                local_dir=download_dir,
                repo_type="dataset",
            )
            print(f"[INFO] Downloaded {file}: {downloaded_path}")
        else:
            print(f"[INFO] {file} already exists in {final_download_dir}")

    # Download variances
    if approximation:
        print("[INFO] Downloading files containing approximated variances.")

        # Download support files
        for file in ["approx_info.txt"]:
            file_path = final_download_dir / file
            if not file_path.exists():
                downloaded_path = hf_hub_download(
                    repo_id="reglab/glove-v",
                    filename=f"{embedding_name}/{file}",
                    local_dir=download_dir,
                    repo_type="dataset",
                )
                print(f"[INFO] Downloaded {file}: {downloaded_path}")
            else:
                print(f"[INFO] {file} already exists in {final_download_dir}")

        # Download approximation files using same pattern as complete files
        chunk_idx = 0
        while True:
            try:
                file_path = final_download_dir / f"approxchunk_{chunk_idx}.safetensors"
                if not file_path.exists():
                    downloaded_path = hf_hub_download(
                        repo_id="reglab/glove-v",
                        filename=f"{embedding_name}/approxchunk_{chunk_idx}.safetensors",
                        local_dir=download_dir,
                        repo_type="dataset",
                    )
                    print(f"[INFO] Downloaded {file_path}")
                chunk_idx += 1
            except Exception:
                # No more chunks to download
                break

    else:
        download_complete_safetensors(
            embedding_name=embedding_name,
            download_dir=download_dir,
        )


def download_complete_safetensors(
    embedding_name: str,
    download_dir: str,
) -> None:
    """
    Downloads chunked safetensor files from HuggingFace.

    Args:
        embedding_name: Name of the corpus on HuggingFace
        download_dir: Path where to save the downloaded chunks
    """
    chunk_idx = 0

    while True:
        try:
            # Download chunk
            file_path = (
                Path(download_dir)
                / embedding_name
                / f"completechunk_{chunk_idx}.safetensors"
            )
            if not file_path.exists():
                _ = hf_hub_download(
                    repo_id="reglab/glove-v",
                    filename=f"{embedding_name}/completechunk_{chunk_idx}.safetensors",
                    local_dir=download_dir,
                    repo_type="dataset",
                )
                print(f"[INFO] Downloaded {file_path}")

            chunk_idx += 1

        except Exception:
            # No more chunks to download
            break

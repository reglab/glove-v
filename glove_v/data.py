"""
This module contains functions to download the GloVe embeddings and GloVe-V variances from HuggingFace. The
download can be either for the approximate variances or the original variances. The approximate variances are
more lightweight, and guarantee 90% reconstruction of the original variance for each word.
"""

import os
from huggingface_hub import hf_hub_download
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file\

import glove_v.utils.file as file_utils


AVAILABLE_EMBEDDINGS = [
    "Toy-Embeddings",
    'COHA_1900-1999_300d',
]

def download_embeddings(
    embedding_name: str,
    approximation: bool = True,
    download_dir: str = f'{file_utils.get_data_path()}/glove-v'
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
        raise ValueError(f"[ERROR] Embeddings should be one of the following: {AVAILABLE_EMBEDDINGS}")

    final_download_dir = os.path.join(download_dir, f"{embedding_name}")
    os.makedirs(final_download_dir, exist_ok=True)

    # Download vocabulary and embeddings
    for file in ['vocab.txt', 'vectors.safetensors']:
        if not os.path.exists(os.path.join(final_download_dir, file)):
            file_path = hf_hub_download(
                repo_id="reglab/glove-v",
                filename=f"{embedding_name}/{file}",
                local_dir=download_dir,
                repo_type="dataset",
            )
            print(f"[INFO] Downloaded {file}: {file_path}")
        else:
            print(f"[INFO] {file} already exists in {final_download_dir}")

    # Download variances
    if approximation:
        print('[INFO] Downloading file containing approximated variances.')
        for file in ['ApproximationVariances.safetensors', 'approx_info.txt']:  
            if not os.path.exists(os.path.join(final_download_dir, file)):
                file_path = hf_hub_download(
                    repo_id="reglab/glove-v",
                    filename=f"{embedding_name}/{file}",
                    local_dir=download_dir,
                    repo_type="dataset",
                )
                print(f"[INFO] Downloaded {file}: {file_path}")
            else:
                print(f"[INFO] {file} already exists in {final_download_dir}")
    else:
        print('[INFO] Downloading file containing complete variances.')
        
        output_path = os.path.join(final_download_dir, "CompleteVariances.safetensors")
        if not os.path.exists(output_path):
            download_and_reconstruct_complete_safetensor(
                embedding_name=embedding_name,
                download_dir=download_dir,
                output_path=output_path,
            )
        else:
            print(f"[INFO] Complete.safetensors already exists in {final_download_dir}")  



def download_and_reconstruct_complete_safetensor(
    embedding_name: str,
    download_dir: str,
    output_path: str,
) -> None:
    """
    Downloads chunked safetensor files from HuggingFace and reconstructs the complete safetensor
    containing the original variances.
    
    Args:
        embedding_name: Name of the corpus on HuggingFace
        download_dir: Path where to save the downloaded chunks
        output_path: Path where to save the reconstructed complete safetensor
    """
    chunk_idx = 0
    all_variances = []
    
    while True:
        try:
            # Download chunk
            chunk_path = hf_hub_download(
                repo_id="reglab/glove-v",
                filename=f"{embedding_name}/complete_chunk_{chunk_idx}.safetensors",
                local_dir=download_dir,
                repo_type="dataset"
            )
            
            # Load chunk data
            with safe_open(chunk_path, framework="numpy") as f:
                # Get variances from chunk
                variances_chunk = f.get_tensor("variances")
                all_variances.append(variances_chunk)
            
            chunk_idx += 1
            
        except Exception as e:
            # No more chunks to download
            break
    
    # Concatenate all variance chunks
    complete_variances = np.concatenate(all_variances, axis=0)
    
    # Save reconstructed complete safetensor
    save_file({
        "variances": complete_variances
    }, output_path)

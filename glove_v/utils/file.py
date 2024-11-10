from pathlib import Path


def get_data_path() -> Path:
    return Path(__file__).parent.parent.parent / "data"


def file_loading_error_message(
    file_name: str, download_dir: str, embedding_name: str
) -> None:
    return print(
        f"[ERROR] {file_name} file not found in {download_dir}/{embedding_name}. "
        "Please make sure you have downloaded the data for these embeddings using the data.download_embeddings function."
    )

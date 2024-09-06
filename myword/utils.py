from pathlib import Path


def get_data_path() -> Path:
    return Path(__file__).parent.parent / "data"

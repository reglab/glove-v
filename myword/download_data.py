import zipfile
from pathlib import Path

import fire
import gdown

import myword.utils

_DEFAULT_ZIP_URL = "https://drive.google.com/file/d/1ho19xLFmgKWWi0xCoKEPV_maKwrhl51k/view?usp=drive_link"
_DEFAULT_OUTPUT_DIR = myword.utils.get_data_path() / "pdf"


def download_gdrive_folder(
    zip_url: str = _DEFAULT_ZIP_URL,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> None:
    """Download a zip file from Google Drive and extract it to a local directory.

    :param zip_url: URL to the folder to download.
    :param output_dir: Local directorty to save the folder to.
    """
    output_dir = Path(output_dir)
    # Name the zip file after the output directory and place it in the same parent directory.
    zip_output_path = output_dir.parent / f"{output_dir.name}.zip"
    gdown.download(zip_url, str(zip_output_path), quiet=False, fuzzy=True)

    # Unzip the folder
    print(f"Unzipping {zip_output_path} to {output_dir}...")
    with zipfile.ZipFile(zip_output_path, "r") as zip_ref:
        # We extract to the output's parent directory to avoid creating a redundant nested directory.
        zip_ref.extractall(output_dir.parent)
    # Delete the zip file
    print(f"Extraction completed. Deleting {zip_output_path}...")
    zip_output_path.unlink()


if __name__ == "__main__":
    fire.Fire(download_gdrive_folder)

"""Train word2vec model from a directory of PDFs"""

import time
from pathlib import Path

import fire
import pdf2image
import pytesseract
import tqdm
from diskcache import Cache

import myword.utils

# This translates to the `data/pdf` directory in the repository root.
PDF_DIR = myword.utils.get_data_path() / "pdf"

pdf_text_cache = Cache(str(myword.utils.get_data_path() / "cache" / "pdf_text"))


def get_text_windows_from_pdf(pdf_path: Path, window_size) -> list[list[str]]:
    if pdf_path in pdf_text_cache:
        text = pdf_text_cache[pdf_path]
    else:
        images = pdf2image.convert_from_path(pdf_path, dpi=200)
        page_texts = [pytesseract.image_to_string(img) for img in images]
        text = "\n".join(page_texts)
        pdf_text_cache[pdf_path] = text

    words = text.lower().split()
    windows = [words[i : i + window_size] for i in range(len(words) - window_size + 1)]
    return windows


def train_model(
    word_windows: list[list[str]],
    output_path: str = "word2vec.model",
    *,
    vector_dim: int = 300,
    num_epochs: int = 10,
) -> None:
    # A dummy function to simulate training a word2vec model.
    for _ in tqdm.trange(num_epochs, desc="Epochs"):
        time.sleep(len(word_windows) * vector_dim * 1e-7)
    print(f"Training complete. Saved model to {output_path}.")


def main(
    pdf_dir: str = str(PDF_DIR),
    output_path: str = "word2vec.model",
    vector_dim: int = 300,
    window_size: int = 5,
    num_epochs: int = 10,
) -> None:
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {pdf_dir}.")
    word_windows = []
    for pdf_file in tqdm.tqdm(pdf_files, desc="PDFs"):
        word_windows.extend(
            get_text_windows_from_pdf(pdf_file, window_size=window_size),
        )
    print(f"Training word2vec model with {len(word_windows)} word windows...")
    train_model(
        word_windows,
        output_path=output_path,
        vector_dim=vector_dim,
        num_epochs=num_epochs,
    )


def entrypoint():
    fire.Fire(main)


if __name__ == "__main__":
    entrypoint()

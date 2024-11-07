# Statistical Uncertainty in Word Embeddings: GloVe-V
**Andrea Vallebueno, Cassandra Handan-Nader, Christopher D. Manning, Daniel E. Ho**

This is the code repository for the paper "Statistical Uncertainty in Word Embeddings: GloVe-V". Our preprint is available [here](https://arxiv.org/abs/2406.12165).

**We introduce a method to obtain approximate, easy-to-use, and scalable uncertainty estimates for the GloVe word embeddings and demonstrate its usefulness in natural language tasks and computational social science analysis. This code repository contains code to download pre-computed GloVe embeddings and GloVe-V variances for several corpora from our HuggingFace repository, to interact with these data products and propagate uncertainty to downstream tasks.**


![GloVe-V](figures/glove_diagram.jpg)

## HuggingFace Repository
xx


## Setup

This project has been tested on Python 3.10, but earlier or later versions may work as well.
First, clone this repo:

```bash
git clone https://github.com/reglab/better-python.git myword
```

Next, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, create a virtual environment:

```bash
cd myword
uv venv  # optionally add --python 3.11 or another version
```

To activate the virtual environment:

```bash
source .venv/bin/activate # If using fish shell, use `source .venv/bin/activate.fish` instead
```

Generally, you will want to activate the virtual environment before running any of the scripts
in this project. However, if you use `uv`, it can handle everything for you! Just run code with
`uv run python <script.py>`. (You can also run anything else installed to your venv, e.g.,
`uv run ruff check --fix`.)

Using `uv run` will also automatically sync your environment with any new/removed packages
in your `pyproject.toml` file. (See the Development section below, or the `uv` docs, for
information about installing/maintaining dependencies.) You can read the `uv` documentation
[here](https://docs.astral.sh/uv/getting-started/features/#projects).

Then, install the dependencies and this package:

```bash
brew install tesseract  # If not on Mac, look up how to install Tesseract for your OS

uv sync
```

Finally, install the git hooks:

```bash
# If you don't already have pre-commit, run: `uv tool install pre-commit`
pre-commit install
```

## Usage

To run the download script:

```bash
python myword/download_data.py
```


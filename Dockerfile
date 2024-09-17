FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV PIP_NO_CACHE_DIR=1

# Install curl/git
RUN apt update -y \
    && apt install -y curl git

# Install Python stuff
WORKDIR /app
COPY pyproject.toml README.md requirements.lock requirements-dev.lock .
RUN pip install -U uv \
    && uv pip install --system transformers pandas datasets tokenizers evaluate more-itertools scikit-learn accelerate wandb click icecream jsonlines pytorch-crf "ray[tune]"

COPY . .

CMD ["python", "src/ner_training/main.py", "train", "--model", "roberta-base-cased", "--definition", "--theorem", "--proof", "--example", "--data_dir", "/volume/data/ner/", "--output_dir", "runs/"]

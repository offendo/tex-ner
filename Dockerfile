FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV PIP_NO_CACHE_DIR=1

# Install curl/git
RUN apt update -y \
    && apt install -y curl git \
    && chmod -R a+w $ELAN_HOME;

# Install Python stuff
WORKDIR /app
COPY pyproject.toml requirements.lock requirements-dev.lock .

RUN pip install -r requirements.lock

#!/usr/bin/env python3

import json
import logging
import math
import os
from pathlib import Path
from pprint import pformat
from typing import Callable, Iterable, Optional

import click
import evaluate
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import wandb
from datasets import Dataset, DatasetDict
from more_itertools import chunked, flatten
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torchcrf import CRF
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    BatchEncoding,
)
from transformers.modeling_outputs import TokenClassifierOutput

from ner_training.utils import create_name_or_ref_tags, PAD_TOKEN_ID, convert_label_to_idx


def load_file_name_or_ref(
    path: str | Path,
    label2id: dict[str, int],
    tokenizer: PreTrainedTokenizer,
    context_len: int = -1,
    strip_bio_prefix: bool = True,
    train_only_tags: list[str] | None = None,
):
    # Load the data
    with open(path, "r") as f:
        data = json.load(f)
    logging.debug(f"Loaded file from {path}.")

    # If we're training on everything, make sure the loader know that
    if train_only_tags is None:
        train_only_tags = list(label2id.keys())

    train_only_ids = [label2id[t] for t in train_only_tags]

    annos = pd.DataFrame.from_records(data["annotations"])
    names = create_name_or_ref_tags(annos, tokenizer, force_reference="reference" in train_only_tags)
    all_examples = []
    for idx, row in names.iterrows():
        tokens = tokenizer(row.tex)
        tags = row.tags

        specials = list(tokenizer.special_tokens_map.values())
        special_tokens = set(map(tokenizer.convert_tokens_to_ids, specials))
        # Add an empty tag for the <s> or </s> tokens
        if tokens.input_ids[0] in special_tokens:
            tags = [[]] + tags
        if tokens.input_ids[-1] in special_tokens:
            tags = tags + [[]]

        if strip_bio_prefix:
            tags = [[t.replace("B-", "").replace("I-", "") for t in tag] for tag in tags]

        # Anything we're not training on, mark it as -100
        tokens["labels"] = [convert_label_to_idx(t, label2id) for t in tags]
        tokens["labels"] = [t if (t in train_only_ids) or (t == 0) else PAD_TOKEN_ID for t in tokens["labels"]]

        # Sanity check to ensure our labels/inputs line up properly
        n_labels = len(tokens["labels"])  # type:ignore
        n_tokens = len(tokens["input_ids"])  # type:ignore
        assert n_labels == n_tokens, f"Mismatch in input/output lengths: {n_labels} == {n_tokens}"

        example = []

        for idx in range(math.ceil(n_tokens / context_len)):
            labels = tokens.labels[idx * context_len : (idx + 1) * context_len]
            input_ids = tokens.input_ids[idx * context_len : (idx + 1) * context_len]
            mask = tokens.attention_mask[idx * context_len : (idx + 1) * context_len]
            all_examples.append({"labels": labels, "input_ids": input_ids, "attention_mask": mask})
    return all_examples


def load_file(
    path: str | Path,
    label2id: dict[str, int],
    tokenizer: PreTrainedTokenizer,
    context_len: int = -1,
    strip_bio_prefix: bool = True,
    examples_as_theorems: bool = False,
    train_only_tags: list[str] | None = None,
):
    if train_only_tags is not None:
        return load_file_name_or_ref(
            path=path,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
            strip_bio_prefix=strip_bio_prefix,
            train_only_tags=train_only_tags,
        )

    # Load the data
    with open(path, "r") as f:
        data = json.load(f)
    logging.debug(f"Loaded file from {path}.")

    iob_tags = data["iob_tags"]  # list of [text, [tags]]
    tex = data["tex"]  # tex string

    # Tokenize the data from the file
    tokens = tokenizer(tex)
    logging.debug(f"Tokenized file into {len(tokens.input_ids)} tokens.")

    tags = [tags for text, tags in iob_tags]
    if strip_bio_prefix:
        tags = [[t.replace("B-", "").replace("I-", "") for t in tag] for tag in tags]
    if examples_as_theorems:
        tags = [list(set([t.replace("example", "theorem") for t in tag])) for tag in tags]

    specials = list(tokenizer.special_tokens_map.values())
    special_tokens = set(map(tokenizer.convert_tokens_to_ids, specials))

    # Add an empty tag for the <s> or </s> tokens
    if tokens.input_ids[0] in special_tokens:
        tags = [[]] + tags
        logging.debug(f"Added an `O` token for the BOS <s> token.")
    if tokens.input_ids[-1] in special_tokens:
        tags = tags + [[]]
        logging.debug(f"Added an `O` token for the EOS </s> token.")

    # Convert the tags to idxs
    tokens["labels"] = [convert_label_to_idx(t, label2id) for t in tags]

    # Sanity check to ensure our labels/inputs line up properly
    n_labels = len(tokens["labels"])  # type:ignore
    n_tokens = len(tokens["input_ids"])  # type:ignore
    assert n_labels == n_tokens, f"Mismatch in input/output lengths: {n_labels} == {n_tokens}"

    # Split it up into context-window sized chunks (for training)
    if context_len > 0:
        sub_examples = []
        for idx in range(math.ceil(n_tokens / context_len)):
            labels = tokens.labels[idx * context_len : (idx + 1) * context_len]
            input_ids = tokens.input_ids[idx * context_len : (idx + 1) * context_len]
            mask = tokens.attention_mask[idx * context_len : (idx + 1) * context_len]
            sub_examples.append({"labels": labels, "input_ids": input_ids, "attention_mask": mask})
        return sub_examples

    return [tokens]


def load_data(
    data_dir: str | Path,
    tokenizer: PreTrainedTokenizer,
    label2id: dict[str, int],
    context_len: int,
    strip_bio_prefix: bool = True,
    examples_as_theorems: bool = False,
    train_only_tags: list[str] | None = None,
):
    logging.info(f"{train_only_tags=}")
    train_dir = Path(data_dir, "train")
    test_dir = Path(data_dir, "test")
    val_dir = Path(data_dir, "val")

    assert train_dir.exists(), f"Expected {train_dir} to exist."
    assert test_dir.exists(), f"Expected {test_dir} to exist."
    assert val_dir.exists(), f"Expected {val_dir} to exist."

    train = []
    for js in os.listdir(train_dir):
        examples = load_file(
            train_dir / js,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
            strip_bio_prefix=strip_bio_prefix,
            examples_as_theorems=examples_as_theorems,
            train_only_tags=train_only_tags,
        )
        train.extend(examples)
    logging.info(f"Loaded train data ({len(train)} examples from {len(os.listdir(train_dir))} files).")

    val = []
    for js in os.listdir(val_dir):
        examples = load_file(
            val_dir / js,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
            strip_bio_prefix=strip_bio_prefix,
            examples_as_theorems=examples_as_theorems,
            train_only_tags=train_only_tags,
        )
        val.extend(examples)
    logging.info(f"Loaded val data ({len(val)} examples from {len(os.listdir(val_dir))} files).")

    test = []
    for js in os.listdir(test_dir):
        examples = load_file(
            test_dir / js,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
            strip_bio_prefix=strip_bio_prefix,
            examples_as_theorems=examples_as_theorems,
            train_only_tags=train_only_tags,
        )
        test.extend(examples)
    logging.info(f"Loaded test data ({len(test)} examples from {len(os.listdir(test_dir))} files).")

    return DatasetDict(
        {
            "train": Dataset.from_list(train),
            "val": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        }
    )

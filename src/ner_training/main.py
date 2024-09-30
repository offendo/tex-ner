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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
metric = evaluate.combine([f1_metric, precision_metric, recall_metric])

PAD_TOKEN_ID = -100

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")


class BertWithCRF(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str | Path,
        label2id: dict[str, int],
        id2label: dict[int, str],
        context_len: int = 512,
        dropout: float = 0.0,
        debug: bool = False,
        crf: bool = False,
    ):
        super().__init__()
        if debug:
            config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=len(label2id))
            config.hidden_size = 128
            config.intermediate_size = 256
            config.num_hidden_layers = 2
            config.num_attention_heads = 2
            self.bert = AutoModelForTokenClassification.from_config(config)
        else:
            self.bert = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name,
                num_labels=len(label2id),
                label2id=label2id,
                id2label=id2label,
                hidden_dropout_prob=dropout,
            )
        self.crf = CRF(len(label2id), batch_first=True) if crf else None
        self.num_labels = len(label2id)
        self.ctx = 512  # this is only used for BERT context window, so just keep it static

    def no_crf_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def crf_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        assert self.crf is not None
        # Handle long documents by first passing everything through BERT and then feeding all at once to the CRF
        B, N = input_ids.shape
        logits = torch.zeros((B, N, self.num_labels), dtype=torch.float32, device=input_ids.device)
        for idx in range(math.ceil(N / 512)):
            outputs = self.bert(
                input_ids=input_ids[:, idx * self.ctx : idx * self.ctx + self.ctx],
                attention_mask=attention_mask[:, idx * self.ctx : idx * self.ctx + self.ctx],
                labels=(
                    labels[:, idx * self.ctx : idx * self.ctx + self.ctx].contiguous() if labels is not None else None
                ),
            )
            logits[:, idx * self.ctx : idx * self.ctx + self.ctx, :] = outputs.logits
        is_pad = labels == -100
        crf_out = self.crf(logits, labels.masked_fill(is_pad, 0), mask=attention_mask.bool(), reduction="token_mean")

        loss = None
        if labels is not None:
            loss = -crf_out

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,  # type:ignore
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        if self.crf:
            return self.crf_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return self.no_crf_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CRFTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, crf: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.bert.device) if class_weights is not None else None
        if class_weights is not None:
            logging.info(f"Using class weights: {class_weights}")
        self.crf = crf

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        if self.crf:
            loss = outputs[0]
            return (loss, outputs) if return_outputs else loss
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def create_multiclass_labels(classes):
    labels = []
    for arg in sorted(classes):
        new = []
        for label in labels:
            new.append(f"{label}-{arg}")
        new.append(arg)
        labels.extend(new)

    labels = ["O"] + labels
    for lab in labels:
        if "name" in lab and "reference" in lab:
            labels.remove(lab)
    return {lab: idx for idx, lab in enumerate(labels)}


def convert_label_to_idx(tags: list[str], label2id: dict[str, int]):
    filtered_tags = sorted([t for t in tags if t in label2id])
    if len(filtered_tags) == 0:
        return label2id["O"]
    return label2id["-".join(filtered_tags)]


def align_annotations_to_tokens(tokens: BatchEncoding, char_tags: list[str]) -> list[list[str]]:
    """Converts character-level annotations to token-level

    Parameters
    ----------
    tokens : BatchEncoding
        Output of a huggingface tokenizer on text
    char_tags : list[list[str]]
        Character-level tags (list of tags per character)

    Returns
    -------
    list[str]
        Token-level tags (list of tags per token)
    """

    aligned_tags: list[list[str]] = []
    for idx in range(len(tokens.input_ids)):
        span = tokens.token_to_chars(idx)
        if span is None:
            continue

        tags_for_token = list(set(char_tags[span.start : span.end]))
        if "O" in tags_for_token and len(tags_for_token) > 1:
            tags_for_token.remove("O")

        # Ensure that we only have B- or I- but not both
        for tag in tags_for_token:
            b = tag.replace("I-", "B-")
            i = tag.replace("B-", "I-")
            if b in tags_for_token and i in tags_for_token:
                tags_for_token.remove(i)
        aligned_tags.append(tags_for_token)
    return aligned_tags


def create_name_or_ref_tags(
    tags: list[str], data: pd.DataFrame, tokenizer: PreTrainedTokenizer, force_reference: bool = True
):
    defs_and_thms = data[data.tag.isin(["theorem", "definition", "example"])]
    records = []
    for idx, outer in defs_and_thms.iterrows():
        inner = data[
            (data.fileid == outer.fileid)
            & (data.start >= outer.start)
            & (data.end <= outer.end)
            & (data.tag.isin(tags))
        ]
        assert isinstance(inner, pd.DataFrame)

        def find_all(text, pattern):
            """Non-regex version of re.findall because I don't want patterns"""
            start = 0
            while True:
                start = text.find(pattern, start)
                if start == -1:
                    return
                yield start
                start += len(pattern)

        # Make sure the references are being repeated
        copies = []
        for idx, ref in inner[inner.tag == "reference"].iterrows():
            if len(ref.text) == 1:
                search_text = f" {ref.text.strip()} "
            else:
                search_text = ref.text
            for start in find_all(outer.text, search_text):
                # We have to offset the start index by the outer start index
                # since we're subtracting the same offset in the next loop
                start = start + outer.start
                end = start + len(ref.text)
                if start == ref.start:
                    continue
                other_tags = inner[
                    ((inner.start <= start) & (start <= inner.end)) | ((inner.start <= end) & (end <= inner.end))
                ]
                if len(other_tags) > 0:
                    continue
                logging.debug(f"Adding a copy of {ref.text}")
                copies.append(
                    dict(
                        start=start + outer.start,
                        end=start + outer.start + len(ref.text),
                        text=ref.text,
                        tag="reference",
                    )
                )
        merged = pd.concat([inner[["start", "end", "text", "tag"]], pd.DataFrame.from_records(copies)])

        if force_reference and (merged.tag == "reference").sum() == 0:
            continue

        # Now we need to merge all the examples
        current_records = []
        tex = outer.text
        tokens = tokenizer(outer.text)
        for i, row in merged.iterrows():
            new_start = row.start - outer.start
            new_end = row.end - outer.start
            row_tags = [
                (f"B-{row.tag}" if i == new_start else f"I-{row.tag}" if (new_start <= i <= new_end) else "O")
                for i in range(len(outer.text))
            ]
            aligned_tags = align_annotations_to_tokens(tokens, row_tags)
            current_records.append(aligned_tags)

        if not current_records:
            continue

        merged_tags = [list(set(flatten(tags_at_idx))) for tags_at_idx in zip(*current_records)]

        records.append({"tex": tex, "tokens": tokens, "tags": merged_tags})

    return pd.DataFrame.from_records(records)


def load_file_name_or_ref(
    path: str | Path,
    name_or_ref: list[str],
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

    annos = pd.DataFrame.from_records(data["annotations"])
    names = create_name_or_ref_tags(name_or_ref, annos, tokenizer, force_reference="reference" in train_only_tags)
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
        tokens["labels"] = [convert_label_to_idx(t, label2id) if t in train_only_tags else PAD_TOKEN_ID for t in tags]

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
    name_or_ref: list[str] | None = None,
    train_only_tags: list[str] | None = None,
):
    if name_or_ref is not None:
        return load_file_name_or_ref(
            path=path,
            name_or_ref=name_or_ref,
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
    name_or_ref: list[str] | None = None,
    train_only_tags: list[str] | None = None,
):
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
            name_or_ref=name_or_ref,
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
            name_or_ref=name_or_ref,
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
            name_or_ref=name_or_ref,
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


def load_model(
    pretrained_model_name: str | Path,
    label2id: dict[str, int],
    debug: bool,
    crf: bool,
    context_len: int,
    dropout: float = 0.0,
    checkpoint: str | Path | None = None,
    randomize_last_layer: bool = False,
    freeze_bert: bool = False,
):
    id2label = {v: k for k, v in label2id.items()}

    model = BertWithCRF(
        pretrained_model_name,
        label2id=label2id,
        id2label=id2label,
        context_len=context_len,
        dropout=dropout,
        debug=debug,
        crf=crf,
    )
    logging.info(f"Loaded BertWithCRF model with base of {pretrained_model_name}")
    if checkpoint is not None:
        state_dict = {}
        from safetensors import safe_open

        with safe_open(Path(checkpoint, "model.safetensors"), framework="pt", device=DEVICE) as file:  # type:ignore
            for k in file.keys():
                state_dict[k] = file.get_tensor(k)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {Path(checkpoint, 'model.safetensors')}")

    # Freeze BERT if needed
    if freeze_bert:
        if hasattr(model.bert, "roberta"):
            bert = model.bert.roberta.encoder
        elif hasattr(model.bert, "bert"):
            bert = model.bert.bert.encoder
        for param in bert.parameters():
            param.requires_grad = False

    # Randomize the last encoder layer, apparently this can help generlization
    if randomize_last_layer:
        last_layer = None
        if hasattr(model.bert, "roberta"):
            last_layer = model.bert.roberta.encoder.layer[-1]
        elif hasattr(model.bert, "bert"):
            last_layer = model.bert.bert.encoder.layer[-1]
        if last_layer is not None:
            for name, param in last_layer.named_parameters():
                if "weight" in name and "LayerNorm" not in name:
                    torch.nn.init.xavier_normal_(param)

    return model


def make_compute_metrics(label2id):
    def compute_metrics(eval_out: EvalPrediction):
        logits, labels = eval_out
        assert isinstance(labels, np.ndarray)

        classes = [cls for cls in label2id.values() if cls != 0]

        preds = np.argmax(logits, axis=-1)
        ls = [l for l in labels.ravel() if l != -100]
        ps = [p for p, l in zip(preds.ravel(), labels.ravel()) if l != -100]

        if len(label2id) == 2:
            p, r, f, _ = precision_recall_fscore_support(ls, ps, average="binary", pos_label=1)
        else:
            p, r, f, _ = precision_recall_fscore_support(ls, ps, average="micro", labels=classes)
        return dict(precision=p, recall=r, f1=f)

    return compute_metrics


@click.group("cli")
def cli():
    pass


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--batch_size", default=8)
@click.option("--learning_rate", default=1e-4)
@click.option("--weight_decay", default=0.0)
@click.option("--steps", default=500)
@click.option("--warmup_ratio", default=0.05)
@click.option("--label_smoothing_factor", default=0.1)
@click.option("--dropout", default=0.0)
@click.option("--scheduler", default="linear")
@click.option("--logging_steps", default=10)
@click.option("--debug", is_flag=True)
@click.option("--use_class_weights", is_flag=True)
@click.option("--randomize_last_layer", is_flag=True)
@click.option("--freeze_bert", is_flag=True)
@click.option("--examples_as_theorems", is_flag=True)
@click.option("--name_or_ref", "-n", type=click.Choice(["name", "reference"]), default=None, multiple=True)
@click.option("--checkpoint", type=click.Path(exists=True, resolve_path=True), default=None)
def train(
    model: str,
    crf: bool,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    context_len: int,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    steps: int,
    warmup_ratio: float,
    label_smoothing_factor: float,
    dropout: float,
    scheduler: str,
    logging_steps: int,
    debug: bool,
    use_class_weights: bool,
    randomize_last_layer: bool,
    freeze_bert: bool,
    examples_as_theorems: bool,
    name_or_ref: list[str] | None,
    checkpoint: Path | None,
):
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name or (name_or_ref and "name" in name_or_ref),
            reference=reference or (name_or_ref and "reference" in name_or_ref),
        ).items()
        if v
    )

    label2id = create_multiclass_labels(class_names)
    logging.info(f"Label map: {label2id}")
    ner_model = load_model(
        model,
        crf=crf,
        context_len=context_len,
        label2id=label2id,
        debug=debug,
        dropout=dropout,
        checkpoint=checkpoint,
        randomize_last_layer=randomize_last_layer,
        freeze_bert=freeze_bert,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(
        data_dir,
        tokenizer,
        context_len=context_len,
        label2id=label2id,
        examples_as_theorems=examples_as_theorems,
        name_or_ref=name_or_ref,
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)

    # Build the trainer
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        max_steps=steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing_factor,
        optim="adamw_torch",
        lr_scheduler_type=scheduler,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_cpu=DEVICE == "cpu",
    )

    if use_class_weights:
        classes = list(label2id.values())
        labels = (
            list(flatten(data["train"]["labels"])) + classes
        )  # add the classes back in so we have at least 1 example
        class_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.array(classes), y=np.array(labels)), dtype=torch.float32
        )
    else:
        class_weights = None

    trainer = CRFTrainer(
        model=ner_model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        compute_metrics=make_compute_metrics(label2id),
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(Path(output_dir) / "checkpoint-best"))


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--overlap_len", default=512, type=int)
@click.option("--batch_size", default=8)
@click.option("--debug", is_flag=True)
@click.option("--examples_as_theorems", is_flag=True)
@click.option("--name_or_ref", "-n", type=click.Choice(["name", "reference"]), default=None, multiple=True)
def test(
    model: str,
    crf: bool,
    checkpoint: Path | None,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    context_len: int,
    overlap_len: int,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    debug: bool,
    examples_as_theorems: bool,
    name_or_ref: str,
):
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name or (name_or_ref and "name" in name_or_ref),
            reference=reference or (name_or_ref and "reference" in name_or_ref),
        ).items()
        if v
    )
    label2id = create_multiclass_labels(class_names)
    logging.info(f"Label map: {label2id}")

    id2label = {v: k for k, v in label2id.items()}
    ner_model = load_model(
        model, crf=crf, context_len=context_len, label2id=label2id, debug=debug, checkpoint=checkpoint, dropout=0.0
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(
        data_dir,
        label2id=label2id,
        tokenizer=tokenizer,
        context_len=context_len,
        examples_as_theorems=examples_as_theorems,
        name_or_ref=name_or_ref,
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        use_cpu=DEVICE == "cpu",
    )
    trainer = CRFTrainer(
        model=ner_model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        compute_metrics=make_compute_metrics(label2id),
    )
    # Run eval on 'test' and 'val'
    for split in ["test", "val"]:
        logits, labels, metrics = trainer.predict(data[split])  # type:ignore
        logging.info(pformat(metrics))
        preds = np.argmax(logits, axis=-1)

        output = {
            "labels": [[id2label[l] for l in ll if l != PAD_TOKEN_ID] for ll in labels],
            "logits": [[p for p, l in zip(pp, ll) if l != PAD_TOKEN_ID] for pp, ll in zip(logits, labels)],
            "preds": [[id2label[p] for p, l in zip(pp, ll) if l != PAD_TOKEN_ID] for pp, ll in zip(preds, labels)],
            "tokens": [[tokenizer.convert_ids_to_tokens(i) for i in item] for item in data[split]["input_ids"]],
        }
        test_df = pd.DataFrame(output)
        test_df.to_json(Path(output_dir, f"{split}.preds.json"))


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--use_class_weights", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--steps", default=500)
@click.option("--logging_steps", default=10)
@click.option("--trials", default=50)
@click.option("--debug", is_flag=True)
@click.option("--examples_as_theorems", is_flag=True)
@click.option("--name_or_ref", "-n", type=click.Choice(["name", "reference"]), default=None, multiple=True)
def tune(
    model: str,
    crf: bool,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    use_class_weights: bool,
    context_len: int,
    data_dir: Path,
    output_dir: Path,
    steps: int,
    trials: int,
    logging_steps: int,
    debug: bool,
    examples_as_theorems: bool,
    name_or_ref: list[str],
):
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name or (name_or_ref and "name" in name_or_ref),
            reference=reference or (name_or_ref and "reference" in name_or_ref),
        ).items()
        if v
    )
    label2id = create_multiclass_labels(class_names)

    def raytune_hp_space(trial):
        return {
            "learning_rate": ray.tune.loguniform(1e-6, 1e-3),
            "per_device_train_batch_size": ray.tune.choice([4, 8, 16, 32]),
            "warmup_ratio": ray.tune.uniform(0.0, 0.1),
            "weight_decay": ray.tune.loguniform(1e-6, 1e-3),
            "lr_scheduler_type": ray.tune.choice(["linear", "cosine", "inverse_sqrt"]),
            "label_smoothing_factor": ray.tune.uniform(0.0, 0.1),
            "dropout": ray.tune.uniform(0.0, 0.5),
            "max_steps": ray.tune.uniform(500, 2000),
        }

    def make_model_init(*args, **kwargs):
        def model_init(trial):
            try:
                dropout = trial["dropout"]
            except Exception as e:
                dropout = 0.0
            return load_model(*args, dropout=dropout, **kwargs)

        return model_init

    def compute_objective(metrics: dict[str, float]):
        return metrics["eval_f1"]

    # Data loading
    tokenizer = AutoTokenizer.from_pretrained(model)
    data = load_data(
        data_dir, tokenizer, context_len=context_len, label2id=label2id, examples_as_theorems=examples_as_theorems
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)
    args = TrainingArguments(
        disable_tqdm=True,
        output_dir=str(output_dir),
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=steps,
        metric_for_best_model="f1",
        use_cpu=DEVICE == "cpu",
    )
    trainer = Trainer(
        model=None,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        compute_metrics=make_compute_metrics(label2id),
        tokenizer=tokenizer,
        model_init=make_model_init(model, label2id=label2id, debug=debug, crf=crf, context_len=context_len),
        data_collator=collator,
    )
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        compute_objective=compute_objective,
        backend="ray",
        hp_space=raytune_hp_space,
        n_trials=trials,
    )
    logging.info("Completed hyperparameter search.")
    logging.info(best_trial)

    save_path = Path(output_dir, "best-trial.pt")
    with open(save_path, "wb") as f:
        torch.save(best_trial, f)
        logging.info(f"Saved hyperparameter search results to {save_path}.")

    return best_trial


if __name__ == "__main__":
    cli.add_command(train)
    cli.add_command(test)
    cli.add_command(tune)
    cli()

#!/usr/bin/env python3

import json
import logging
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import click
import numpy as np
import pandas as pd
import safetensors
import torch
import torch.nn as nn
import wandb
import evaluate
from sklearn.metrics import precision_recall_fscore_support
from more_itertools import flatten
from datasets import Dataset, DatasetDict
from more_itertools import chunked
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

os.environ["TOKENIZERS_PARALLELISM"] = "false"

f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
metric = evaluate.combine([f1_metric, precision_metric, recall_metric])

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")


class BertWithCRF(nn.Module):
    def __init__(
        self, pretrained_model_name: str, num_labels: int, context_len: int = 512
    ):
        super().__init__()
        from torchcrf import CRF

        self.bert = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, num_labels=num_labels
        )
        self.num_labels = num_labels
        self.crf = CRF(num_labels, batch_first=True)
        self.ctx = context_len

    def decode(self, batch):
        # Handle long documents by first passing everything through BERT and then feeding all at once to the CRF
        B, N = batch["input_ids"].shape
        batch_outputs = torch.zeros(
            (B, N, self.num_labels), dtype=torch.float32, device=self.bert.device
        )
        batch_inputs = batch["input_ids"]
        batch_mask = batch["attention_mask"]
        for idx in range(math.ceil(N / 512)):
            input_ids = batch_inputs[:, idx * self.ctx : idx * self.ctx + self.ctx]
            mask = batch_mask[:, idx * self.ctx : idx * self.ctx + self.ctx]
            outputs = self.bert(
                input_ids=input_ids.to(model.bert.device),
                attention_mask=mask.to(model.bert.device),
            )
            batch_outputs[:, idx * self.ctx : idx * self.ctx + self.ctx, :] = (
                outputs.logits
            )
        crf_out = self.crf.decode(batch_outputs, mask=batch_mask.bool())
        return crf_out

    def forward(
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
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        return_dict = True
        # Handle long documents by first passing everything through BERT and then feeding all at once to the CRF
        B, N = input_ids.shape
        logits = torch.zeros(
            (B, N, self.num_labels), dtype=torch.float32, device=input_ids.device
        )
        for idx in range(math.ceil(N / 512)):
            outputs = self.bert(
                input_ids=input_ids[:, idx * self.ctx : idx * self.ctx + self.ctx],
                attention_mask=attention_mask[
                    :, idx * self.ctx : idx * self.ctx + self.ctx
                ],
                labels=labels[
                    :, idx * self.ctx : idx * self.ctx + self.ctx
                ].contiguous(),
            )
            logits[:, idx * self.ctx : idx * self.ctx + self.ctx, :] = outputs.logits
        crf_out = self.crf(
            logits, labels, mask=attention_mask.bool(), reduction="token_mean"
        )

        loss = None
        if labels is not None:
            loss = -crf_out

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,  # type:ignore
            hidden_states=None,
            attentions=None,
        )


def create_multiclass_labels(classes):
    labels = []
    for arg in sorted(classes):
        new = []
        for label in labels:
            new.append(f"{label}-{arg}")
        new.append(arg)
        labels.extend(new)

    labels = ["O"] + labels
    return {lab: idx for idx, lab in enumerate(labels)}


def convert_label_to_idx(tags: list[str], label2id: dict[str, int]):
    filtered_tags = sorted([t for t in tags if t in label2id])
    if len(filtered_tags) == 0:
        return label2id["O"]
    return label2id["-".join(filtered_tags)]


def _load_file(
    path: str | Path,
    label2id: dict[str, int],
    tokenizer: PreTrainedTokenizer,
    context_len: Optional[int] = None,
    strip_bio_prefix: bool = True,
):
    # Load the data
    with open(path, "r") as f:
        data = json.load(f)
    logging.debug(f"Loaded file from {path}.")

    iob_tags = data["iob_tags"]  # list of [text, [tags]]
    tex = data["tex"]  # tex string

    # Tokenize the data from the file
    tokens = tokenizer(tex)
    logging.debug(f"Tokenized file into {len(tokens)} tokens.")

    tags = [tags for text, tags in iob_tags]
    if strip_bio_prefix:
        tags = [[t.replace("B-", "").replace("I-", "") for t in tag] for tag in tags]

    special_tokens = set(
        map(tokenizer.convert_tokens_to_ids, tokenizer.special_tokens_map.values())
    )

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
    assert (
        n_labels == n_tokens
    ), f"Mismatch in input/output lengths: {n_labels} == {n_tokens}"

    # Split it up into context-window sized chunks (for training)
    if context_len is not None:
        sub_examples = []
        for idx in range(math.ceil(n_tokens / context_len)):
            labels = tokens["labels"][idx * context_len : (idx + 1) * context_len]
            input_ids = tokens["input_ids"][idx * context_len : (idx + 1) * context_len]
            mask = tokens["attention_mask"][idx * context_len : (idx + 1) * context_len]
            sub_examples.append(
                {"labels": labels, "input_ids": input_ids, "attention_mask": mask}
            )
        return sub_examples

    return [tokens]


def load_data(
    data_dir: str | Path,
    tokenizer: PreTrainedTokenizer,
    label2id: dict[str, int],
    context_len: int,
    strip_bio_prefix: bool = True,
):
    train_dir = Path(data_dir, "train")
    test_dir = Path(data_dir, "test")
    val_dir = Path(data_dir, "val")

    assert train_dir.exists(), f"Expected {train_dir} to exist."
    assert test_dir.exists(), f"Expected {test_dir} to exist."
    assert val_dir.exists(), f"Expected {val_dir} to exist."

    train = []
    for js in os.listdir(train_dir):
        examples = _load_file(
            train_dir / js,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
        )
        train.extend(examples)
    logging.info(
        f"Loaded train data ({len(train)} examples from {len(os.listdir(train_dir))} files)."
    )

    val = []
    for js in os.listdir(val_dir):
        examples = _load_file(
            val_dir / js,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
        )
        val.extend(examples)
    logging.info(
        f"Loaded val data ({len(val)} examples from {len(os.listdir(val_dir))} files)."
    )

    test = []
    for js in os.listdir(test_dir):
        examples = _load_file(
            test_dir / js,
            label2id=label2id,
            tokenizer=tokenizer,
            context_len=context_len,
        )
        test.extend(examples)
    logging.info(
        f"Loaded test data ({len(test)} examples from {len(os.listdir(test_dir))} files)."
    )

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
    checkpoint: str | Path | None = None,
):
    id2label = {v: k for k, v in label2id.items()}

    # Shrink the size if we're debugging stuff
    if debug:
        config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        model = AutoModelForTokenClassification.from_config(config).to(DEVICE)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        ).to(DEVICE)

    logging.info(f"Loaded MultiLabelNER model with base of {pretrained_model_name}")
    if checkpoint is not None:
        state_dict = {}
        with safetensors.safe_open(
            Path(checkpoint, "model.safetensors"), framework="pt", device=DEVICE
        ) as file:
            for k in file.keys():
                state_dict[k] = file.get_tensor(k)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {Path(checkpoint, 'model.safetensors')}")
    return model


def compute_metrics(eval_out: EvalPrediction):
    logits, labels = eval_out
    preds = np.argmax(logits, axis=-1)
    metrics = metric.compute(
        references=labels.ravel(),
        predictions=preds.ravel(),
        labels=list(range(1, np.max(labels))),
        average="weighted",
    )
    return metrics


@click.group("cli")
def cli():
    pass


@click.command()
@click.option("--model", type=str)
@click.option(
    "--data_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    "--output_dir",
    type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True),
)
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--batch_size", default=8)
@click.option("--learning_rate", default=1e-4)
@click.option("--epochs", default=15)
@click.option("--warmup_ratio", default=0.05)
@click.option("--label_smoothing_factor", default=0.1)
@click.option("--logging_steps", default=10)
@click.option("--debug", is_flag=True)
def train(
    model: str,
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
    epochs: int,
    warmup_ratio: float,
    label_smoothing_factor: float,
    logging_steps: int,
    debug: bool,
):
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name,
            reference=reference,
        ).items()
        if v
    )
    label2id = create_multiclass_labels(class_names)
    logging.info(f"Label map: {label2id}")
    ner_model = load_model(model, label2id=label2id, debug=debug)
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(data_dir, tokenizer, context_len=context_len, label2id=label2id)
    collator = DataCollatorForTokenClassification(
        tokenizer, padding=True, label_pad_token_id=-100
    )

    # Build the trainer
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        use_cpu=DEVICE == "cpu",
    )
    trainer = Trainer(
        model=ner_model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(Path(output_dir) / "checkpoint-final"))


@click.command()
@click.option("--model", type=str)
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True),
)
@click.option(
    "--output_dir",
    type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True),
)
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
def test(
    model: str,
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
):
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name,
            reference=reference,
        ).items()
        if v
    )
    label2id = create_multiclass_labels(class_names)
    id2label = {v: k for k, v in label2id.items()}
    ner_model = load_model(model, label2id=label2id, debug=debug, checkpoint=checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(
        data_dir, label2id=label2id, tokenizer=tokenizer, context_len=context_len
    )
    collator = DataCollatorForTokenClassification(
        tokenizer, padding=True, label_pad_token_id=-100
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        use_cpu=DEVICE == "cpu",
    )
    trainer = Trainer(
        model=ner_model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        compute_metrics=compute_metrics,
    )
    logits, labels, metrics = trainer.predict(data["test"])  # type:ignore
    preds = np.argmax(logits, axis=-1)

    output = {
        "labels": [[id2label[l] for l in ll if l != -100] for ll in labels],
        "preds": [
            [id2label[p] for p, l in zip(pp, ll) if l != -100]
            for pp, ll in zip(preds, labels)
        ],
        "tokens": [
            [tokenizer.convert_ids_to_tokens(i) for i in item]
            for item in data["test"]["input_ids"]
        ],
    }
    test_df = pd.DataFrame(output)
    test_df.to_json(Path(output_dir, "preds.json"))


if __name__ == "__main__":
    cli.add_command(train)
    cli.add_command(test)
    cli()

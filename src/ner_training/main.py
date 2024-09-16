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
from datasets import Dataset, DatasetDict
from more_itertools import chunked
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")


def _load_file(
    path: str | Path,
    classes: Iterable[str],
    tokenizer: PreTrainedTokenizer,
    context_len: Optional[int] = None,
    strip_bio_prefix: bool = True,
):
    # Train the MLB on nothing because we already know the classes
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([])
    logging.info(f"Initialized multilabel binarizer with classes: {classes}.")

    # Load the data
    with open(path, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded file from {path}.")

    iob_tags = data["iob_tags"]  # list of [text, [tags]]
    tex = data["tex"]  # tex string

    # Tokenize the data from the file
    tokens = tokenizer(tex)
    logging.info(f"Tokenized file into {len(tokens)} tokens.")

    tags = [tags for text, tags in iob_tags]
    if strip_bio_prefix:
        tags = [[t.replace("B-", "").replace("I-", "") for t in tag] for tag in tags]
    special_tokens = set(map(tokenizer.convert_tokens_to_ids, tokenizer.special_tokens_map.values()))

    # Add an empty tag for the <s> or </s> tokens
    if tokens.input_ids[0] in special_tokens:
        tags = [[]] + tags
        logging.info(f"Added an `O` token for the BOS <s> token.")
    if tokens.input_ids[-1] in special_tokens:
        tags = tags + [[]]
        logging.info(f"Added an `O` token for the EOS </s> token.")

    tokens["labels"] = mlb.transform(tags)

    # Sanity check to ensure our labels/inputs line up properly
    n_labels = len(tokens["labels"])  # type:ignore
    n_tokens = len(tokens["input_ids"])  # type:ignore
    assert n_labels == n_tokens, f"Mismatch in input/output lengths: {n_labels} == {n_tokens}"

    # Split it up into context-window sized chunks (for training)
    if context_len is not None:
        sub_examples = []
        for idx in range(math.ceil(n_tokens / context_len)):
            labels = tokens["labels"][idx * context_len : (idx + 1) * context_len]
            input_ids = tokens["input_ids"][idx * context_len : (idx + 1) * context_len]
            mask = tokens["attention_mask"][idx * context_len : (idx + 1) * context_len]
            sub_examples.append({"labels": labels, "input_ids": input_ids, "attention_mask": mask})
        return sub_examples

    return [tokens]


def load_data(
    data_dir: str | Path,
    tokenizer: PreTrainedTokenizer,
    classes: Iterable[str],
    context_len: int,
    strip_bio_prefix: bool = True,
):
    train_dir = Path(data_dir, "train")
    test_dir = Path(data_dir, "test")

    assert train_dir.exists(), f"Expected {train_dir} to exist."
    assert test_dir.exists(), f"Expected {test_dir} to exist."

    train = []
    for js in os.listdir(train_dir):
        examples = _load_file(train_dir / js, classes=classes, tokenizer=tokenizer, context_len=context_len)
        train.extend(examples)
    logging.info(f"Loaded train data ({len(train)} examples from {len(os.listdir(train_dir))} files).")

    test = []
    for js in os.listdir(test_dir):
        examples = _load_file(test_dir / js, classes=classes, tokenizer=tokenizer, context_len=context_len)
        test.extend(examples)
    logging.info(f"Loaded test data ({len(test)} examples from {len(os.listdir(test_dir))} files).")

    return DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})


class MultiLabelNERTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(DEVICE)
            logging.info(f"Using multi-label classification with class weights", class_weights)
        self.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # this accesses predictions for tokens that aren't CLS, PAD, or the 2nd+ subword in a word
        # and simultaneously flattens the logits or labels
        flat_outputs = outputs.logits[labels != -100]
        flat_labels = labels[labels != -100]

        loss = self.loss_fct(flat_outputs, flat_labels.float())

        return (loss, outputs) if return_outputs else loss


class MultiLabelNER(nn.Module):
    def __init__(
        self, pretrained_model_name, num_labels: int, class_weights: Optional[torch.Tensor] = None, debug: bool = False
    ):
        super().__init__()
        self.num_labels = num_labels
        if debug:
            bert_config = AutoConfig.from_pretrained(pretrained_model_name)
            bert_config.hidden_size = 128
            bert_config.intermediate_size = 256
            bert_config.num_hidden_layers = 2
            bert_config.num_attention_heads = 2
            self.bert = AutoModel.from_config(bert_config)
        else:
            self.bert = AutoModel.from_pretrained(
                pretrained_model_name,
                # attn_implementation="flash_attention_2",
                num_labels=num_labels,
            )
        self.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights)
        self.head = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

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

        return_dict = return_dict if return_dict is not None else True

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.head(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_model(pretrained_model_name: str | Path, num_labels: int, debug: bool, checkpoint: str | Path | None = None):
    model = MultiLabelNER(pretrained_model_name, num_labels=num_labels, debug=debug).to(DEVICE)
    logging.info(f"Loaded MultiLabelNER model with base of {pretrained_model_name}")
    if checkpoint is not None:
        state_dict = {}
        with safetensors.safe_open(Path(checkpoint, "model.safetensors"), framework="pt", device=DEVICE) as file:
            for k in file.keys():
                state_dict[k] = file.get_tensor(k)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {Path(checkpoint, 'model.safetensors')}")
    return model


def predict(
    model: MultiLabelNER,
    data: Dataset,
    collator: Callable,
    ctx_len: int = 512,
    overlap: int = 512,
    batch_size: int = 8,
):
    all_predictions = []
    for examples in chunked(data, n=batch_size):
        batch = collator(examples)
        batch_size, n_tokens = batch["input_ids"].size()

        # Batch container for (1) logits and (2) number of times we overlap a particular index
        batch_logits = torch.zeros((batch_size, n_tokens, model.num_labels), dtype=torch.float)
        batch_counts = torch.zeros((n_tokens,), dtype=torch.long)
        for idx in range(math.ceil(n_tokens / overlap)):
            # Run the model
            input_ids = batch["input_ids"][idx * overlap : idx * overlap + ctx_len]
            mask = batch["attention_mask"][idx * overlap : idx * overlap + ctx_len]
            labels = batch["labels"][idx * overlap : idx * overlap + ctx_len]
            outputs = model(
                input_ids=input_ids.to(DEVICE),
                labels=labels.to(DEVICE).float(),
                attention_mask=mask.to(DEVICE),
            )
            logits = outputs.logits.detach().cpu()

            # Add the logits to the previous predictions, and increment the index count
            batch_logits[:, idx * overlap : idx * overlap + ctx_len, :] += logits
            batch_counts[idx * overlap : idx * overlap + ctx_len] += 1

        # Every set of logits in the middle got over-counted by overlap, so we have to average them out.
        batch_logits = batch_logits / batch_counts.view(1, -1, 1)
        batch_predictions = (torch.sigmoid(batch_logits) >= 0.5).long().numpy()
        all_predictions.append(batch_predictions)
    return all_predictions


@click.group("cli")
def cli():
    pass


@click.command()
@click.option("--model", type=str)
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
    classes = tuple(
        k
        for k, v in dict(
            definition=definition, theorem=theorem, proof=proof, example=example, name=name, reference=reference
        ).items()
        if v
    )
    ner_model = load_model(model, num_labels=len(classes), debug=debug)
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(data_dir, tokenizer, context_len=context_len, classes=classes)
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=[-100] * len(classes))

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
        save_strategy="epoch",
        save_total_limit=3,
        use_cpu=DEVICE == "cpu",
    )
    trainer = MultiLabelNERTrainer(model=ner_model, args=args, data_collator=collator, train_dataset=data["train"])
    trainer.train()


@click.command()
@click.option("--model", type=str)
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option("--output_file", type=click.File("w"))
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
    output_file: Path,
    batch_size: int,
    debug: bool,
):
    classes = tuple(
        k
        for k, v in dict(
            definition=definition, theorem=theorem, proof=proof, example=example, name=name, reference=reference
        ).items()
        if v
    )
    ner_model = load_model(model, num_labels=len(classes), debug=debug, checkpoint=checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model)
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([])

    # Data loading
    data = load_data(data_dir, classes=classes, tokenizer=tokenizer, context_len=context_len)
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=[-100] * len(classes))

    outputs = predict(
        model=ner_model,
        data=data["test"],
        collator=collator,
        ctx_len=context_len,
        overlap=overlap_len,
        batch_size=batch_size,
    )
    preds = [mlb.inverse_transform(example) for batch in outputs for example in batch]  # S, N, T
    labels = [mlb.inverse_transform(np.array(example)) for example in data["test"]["labels"]]
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in data["test"]["input_ids"]]
    test_df = pd.DataFrame(dict(preds=preds, labels=labels, tokens=tokens))
    test_df.to_json(output_file)


if __name__ == "__main__":
    cli.add_command(train)
    cli.add_command(test)
    cli()

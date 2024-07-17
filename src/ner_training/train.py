#!/usr/bin/env python3

import json
import logging
import os
import math
from typing import Optional
from pathlib import Path
from more_itertools import chunked

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    AutoModel,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput

CLASSES = ["definition", "theorem", "name", "reference", "proof", "example"]


def _load_file(path, tokenizer, ctx_len: Optional[int] = None):
    mlb = MultiLabelBinarizer(classes=CLASSES)
    mlb.fit([])
    with open(path, "r") as f:
        data = json.load(f)
    iob_tags = data["iob_tags"]  # list of [text, [tags]]
    tex = data["tex"]  # tex string

    tokens = tokenizer(tex)
    tags = [tags for text, tags in iob_tags]
    special_tokens = set(map(tokenizer.convert_tokens_to_ids, tokenizer.special_tokens_map.values()))

    # Add an 'O' for the <s> or </s> tokens
    if tokens.input_ids[0] in special_tokens:
        tags = ["O"] + tags
    if tokens.input_ids[-1] in special_tokens:
        tags = tags + ["O"]
    tokens["labels"] = mlb.transform(tags)

    # Sanity check to ensure our labels/inputs line up properly
    n_labels = len(tokens["labels"])  # type:ignore
    n_tokens = len(tokens["input_ids"])  # type:ignore
    assert n_labels == n_tokens, f"Mismatch in input/output lengths: {n_labels} == {n_tokens}"

    # Split it up into context-window sized chunks (for training)
    if ctx_len is not None:
        sub_examples = []
        for idx in range(math.ceil(n_tokens / ctx_len)):
            labels = tokens["labels"][idx * ctx_len : (idx + 1) * ctx_len]
            input_ids = tokens["input_ids"][idx * ctx_len : (idx + 1) * ctx_len]
            mask = tokens["attention_mask"][idx * ctx_len : (idx + 1) * ctx_len]
            sub_examples.append({"labels": labels, "input_ids": input_ids, "attention_mask": mask})
        return sub_examples

    return [tokens]


def load_data(data_dir: str, tokenizer: PreTrainedTokenizer, max_ctx_len):
    train_dir = Path(data_dir, "train")
    test_dir = Path(data_dir, "test")

    assert train_dir.exists(), f"Expected {train_dir} to exist."
    assert test_dir.exists(), f"Expected {test_dir} to exist."

    train = []
    for js in os.listdir(train_dir):
        examples = _load_file(train_dir / js, tokenizer, max_ctx_len)
        train.extend(examples)

    test = []
    for js in os.listdir(test_dir):
        examples = _load_file(test_dir / js, tokenizer)
        test.extend(examples)

    return DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})


class MultiLabelNERTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.FloatTensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
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

        try:
            loss = self.loss_fct(flat_outputs, flat_labels.float())
        except AttributeError:  # DataParallel
            loss = self.loss_fct(flat_outputs, flat_labels.float())

        return (loss, outputs) if return_outputs else loss


class MultiLabelNER(nn.Module):
    def __init__(self, pretrained_model_name, num_labels: int, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_labels = num_labels
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


def collate(examples):
    pass


def load_model(pretrained_model_name: str):
    model = MultiLabelNER(pretrained_model_name, num_labels=len(CLASSES))
    return model.to("mps")


if __name__ == "__main__":
    MODEL = "bert-base-cased"
    model = load_model(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = load_data("./data/", tokenizer, max_ctx_len=512)

    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=[-100] * len(CLASSES))
    args = TrainingArguments(
        output_dir="./runs/",
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_strategy="steps",
        logging_steps=2,
        num_train_epochs=15,
        warmup_ratio=0.05,
        save_strategy="steps",
        save_total_limit=3,
        label_smoothing_factor=0.1,
        optim="adamw_torch",
        use_cpu=True,
    )
    trainer = MultiLabelNERTrainer(model=model, args=args, data_collator=collator, train_dataset=data["train"])
    trainer.train()

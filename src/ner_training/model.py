#!/usr/bin/env python3
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn.utils.rnn import pad_sequence

from ner_training.utils import PAD_TOKEN_ID


@dataclass
class CRFOutput(TokenClassifierOutput):
    predictions: torch.Tensor | None = None


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
        return_predictions: Optional[bool] = None,
    ) -> CRFOutput:
        assert self.crf is not None
        # Handle long documents by first passing everything through BERT and then feeding all at once to the CRF
        B, N = input_ids.shape
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None,
        )
        loss = None
        if labels is not None:
            is_pad = labels == -100
            crf_out = self.crf.forward(
                outputs.logits, labels.masked_fill(is_pad, 0), mask=attention_mask.bool(), reduction="mean"
            )
            loss = -crf_out

        if return_predictions or not self.training:
            preds = pad_sequence(
                [torch.tensor(t) for t in self.crf.decode(outputs.logits, mask=attention_mask.bool())],
                padding_value=PAD_TOKEN_ID,
                batch_first=True,
            )
        else:
            preds = None

        return CRFOutput(
            loss=loss,
            logits=outputs.logits,  # type:ignore
            hidden_states=None,
            attentions=None,
            predictions=preds,
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
        return_predictions: Optional[bool] = None,
    ) -> CRFOutput:
        if self.crf is not None:
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
                return_predictions=return_predictions,
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


class StackedBERTWithCRF(nn.Module):
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
        self.base = BertWithCRF(
            pretrained_model_name,
            label2id=label2id,
            id2label=id2label,
            context_len=context_len,
            dropout=dropout,
            crf=crf,
            debug=debug,
        )
        self.tag_pad_token = len(label2id)

        # Freeze the base model
        # for param in self.base.parameters():
        #     param.requires_grad = False

        if debug:
            self.tag_embedding = nn.Embedding(
                num_embeddings=len(label2id) + 1, embedding_dim=128, padding_idx=self.tag_pad_token
            )
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=256, dropout=dropout)
            self.stack = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(128, len(label2id))
        else:
            self.tag_embedding = nn.Embedding(
                num_embeddings=len(label2id) + 1, embedding_dim=512, padding_idx=self.tag_pad_token
            )
            layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=dropout)
            self.stack = nn.TransformerEncoder(layer, num_layers=4)
            self.head = nn.Linear(512, len(label2id))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
        self.num_labels = len(label2id)
        self.ctx = 512  # this is only used for BERT context window, so just keep it static

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        bert_output = self.base.forward(
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
            return_predictions=True,
        )

        bert_preds = bert_output.predictions.masked_fill(bert_output.predictions == PAD_TOKEN_ID, self.tag_pad_token)
        x = self.tag_embedding(bert_preds)
        x = self.stack.forward(x, src_key_padding_mask=(bert_preds == self.tag_pad_token).T)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            loss += bert_output.loss

        return TokenClassifierOutput(
            loss=bert_output.loss,
            logits=bert_output.logits,
            hidden_states=None,
            attentions=None,
        )

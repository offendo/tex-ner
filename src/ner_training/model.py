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
    ) -> CRFOutput:
        assert self.crf is not None
        # Handle long documents by first passing everything through BERT and then feeding all at once to the CRF
        B, N = input_ids.shape
        logits = torch.zeros((B, N, self.num_labels), dtype=torch.float32, device=input_ids.device)
        for idx in range(math.ceil(N / 512)):
            start = idx * self.ctx
            end = idx * self.ctx + self.ctx
            outputs = self.bert(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask[:, start:end],
                labels=labels[:, start:end].contiguous() if labels is not None else None,
            )
            logits[:, start:end, :] = outputs.logits

        loss = None
        if labels is not None:
            is_pad = labels == -100
            crf_out = self.crf(
                logits, labels.masked_fill(is_pad, 0), mask=attention_mask.bool(), reduction="token_mean"
            )
            loss = -crf_out

        if not self.training:
            preds = pad_sequence(
                [torch.tensor(t) for t in self.crf.decode(logits, mask=attention_mask.bool())],
                padding_value=PAD_TOKEN_ID,
                batch_first=True,
            )
        else:
            preds = None

        return CRFOutput(
            loss=loss,
            logits=logits,  # type:ignore
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
    ) -> CRFOutput:
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

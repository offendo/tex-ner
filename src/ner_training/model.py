#!/usr/bin/env python3
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from more_itertools import windowed
from transformers import AutoConfig, AutoModelForTokenClassification, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn.utils.rnn import pad_sequence

from ner_training.crf import CRF, SemiCRF
from ner_training.utils import PAD_TOKEN_ID


@dataclass
class BertWithCRFConfig:
    model_name_or_path: str | Path
    label2id: dict[str, int]
    id2label: dict[int, str]

    # Defaults
    context_len: int = 512
    overlap_len: int = 512
    dropout: float = 0.0
    debug: bool = False
    crf_loss_reduction: str = "token_mean"
    crf_segment_length: int = 1


@dataclass
class CRFOutput(TokenClassifierOutput):
    predictions: torch.Tensor | None = None


class BertWithCRF(PreTrainedModel):
    def __init__(self, config: BertWithCRFConfig):
        bert_config = AutoConfig.from_pretrained(config.model_name_or_path, num_labels=len(config.label2id))
        super().__init__(bert_config)
        self.num_labels = len(config.label2id)
        self.crf_loss_reduction = config.crf_loss_reduction
        self.ctx = config.context_len
        self.overlap = config.overlap_len
        if config.debug:
            bert_config.hidden_size = 128
            bert_config.intermediate_size = 256
            bert_config.num_hidden_layers = 2
            bert_config.num_attention_heads = 2
            self.bert = AutoModelForTokenClassification.from_config(bert_config)
        else:
            self.bert = AutoModelForTokenClassification.from_pretrained(
                config.model_name_or_path,
                num_labels=len(config.label2id),
                label2id=config.label2id,
                id2label=config.id2label,
                hidden_dropout_prob=config.dropout,
            )
        if config.crf_segment_length == 1:
            self.crf = CRF(self.num_labels, batch_first=True)
        elif config.crf_segment_length > 1:
            self.crf = SemiCRF(
                self.num_labels,
                batch_first=True,
                padding_idx=-100,
                max_segment_length=config.crf_segment_length,
            )
        else:
            self.crf = None

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
        logits = torch.zeros((B, N, self.num_labels), dtype=torch.float32, device=input_ids.device)
        counts = torch.zeros(N, dtype=torch.int, device=input_ids.device)

        for idx in range(math.ceil(N / self.overlap)):
            start = idx * self.overlap
            end = idx * self.overlap + self.ctx
            outputs = self.bert(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask[:, start:end],
                labels=labels[:, start:end].contiguous() if labels is not None else None,
            )
            logits[:, start:end, :] += outputs.logits
            counts[start:end] += 1

        # Average them out
        logits = logits / counts.view(1, -1, 1)

        loss = None
        if labels is not None:
            is_pad = labels == -100
            crf_out = self.crf.forward(
                logits,
                tags=labels.masked_fill(is_pad, 0),
                mask=attention_mask.bool(),
                reduction=self.crf_loss_reduction,
            )
            loss = -crf_out

        if return_predictions or not self.training:
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
            hidden_states=outputs.hidden_states,
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


class StackedBertWithCRF(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str | Path,
        label2id: dict[str, int],
        id2label: dict[int, str],
        context_len: int = 512,
        overlap_len: int = 512,
        dropout: float = 0.0,
        debug: bool = False,
        use_input_ids: bool = False,
        crf_loss_reduction: str = "token_mean",
        crf_segment_length: int = 1,
    ):
        super().__init__()
        self.base = BertWithCRF(
            pretrained_model_name,
            label2id=label2id,
            id2label=id2label,
            context_len=context_len,
            overlap_len=overlap_len,
            dropout=dropout,
            debug=debug,
            crf_loss_reduction=crf_loss_reduction,
            crf_segment_length=crf_segment_length,
        )
        self.tag_pad_token = 0
        self.use_input_ids = use_input_ids

        # Freeze the base model
        for param in self.base.parameters():
            param.requires_grad = False

        if debug:
            config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=len(label2id))
            config.vocab_size = len(label2id) + 1
            config.hidden_size = 128
            config.intermediate_size = 256
            config.num_hidden_layers = 2
            config.num_attention_heads = 2
            config.pad_token_id = self.tag_pad_token
            config.bos_token_id = None
            config.eos_token_id = None
            self.tagger = AutoModelForTokenClassification.from_config(config)
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=len(label2id))
            config.vocab_size = len(label2id) + 1
            config.pad_token_id = self.tag_pad_token
            config.bos_token_id = None
            config.eos_token_id = None
            self.tagger = AutoModelForTokenClassification.from_config(config)

        self.num_labels = len(label2id)
        self.ctx = 512  # this is only used for BERT context window, so just keep it static

    def forward(
        self,
        input_ids: torch.Tensor,
        tag_ids: torch.Tensor,
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
        # bert_output = self.base.forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     labels=labels,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     return_predictions=True,
        # )

        tag_embeddings = self.tagger.roberta.embeddings(
            input_ids=tag_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # Create the input embeds as a sum of the encoded tokens + embedded tags
        if self.use_input_ids:
            # Get the input token embeddings
            encodings = self.base.bert.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            input_embeds = encodings.hidden_states[-1] + tag_embeddings
        else:
            input_embeds = tag_embeddings

        # Shift things up by 1 so we can use 0 for padding
        # bert_preds = (
        #     (bert_output.predictions + 1).to(input_ids.device).masked_fill(~attention_mask.bool(), self.tag_pad_token)
        # )

        tagger_output = self.tagger.forward(
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        loss = None
        if labels is not None:
            loss = tagger_output.loss
            logits = tagger_output.logits

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

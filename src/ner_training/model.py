#!/usr/bin/env python3
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn.utils.rnn import pad_sequence

from ner_training.crf import CRF
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
        crf_loss_reduction: str = "token_mean",
        add_second_max_to_o: Optional[bool] = None,
        use_crf_cost_function: bool = False,
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
        self.add_second_max_to_o = add_second_max_to_o
        self.crf_loss_reduction = crf_loss_reduction
        self.crf = CRF(len(label2id), batch_first=True, use_cost_function=use_crf_cost_function) if crf else None
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
            if self.add_second_max_to_o:
                values, indices = torch.topk(outputs.logits[:, :, 1:], dim=-1, k=2)
                outputs.logits[:, :, 0] += values[:, :, 1]
            is_pad = labels == -100
            crf_out = self.crf.forward(
                outputs.logits,
                labels.masked_fill(is_pad, 0),
                mask=attention_mask.bool(),
                reduction=self.crf_loss_reduction,
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
        self.tag_pad_token = 0

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

        tag_embeddings = self.tagger.roberta.embeddings(
            input_ids=tag_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # Create the input embeds as a sum of the encoded tokens + embedded tags
        input_embeds = encodings.hidden_states[-1] + tag_embeddings

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

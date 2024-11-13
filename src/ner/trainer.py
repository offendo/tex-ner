#!/usr/bin/env python3
import logging

import torch
import torch.nn as nn
from transformers import Trainer


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
        if hasattr(outputs, "loss"):
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

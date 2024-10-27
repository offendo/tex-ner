#!/usr/bin/env python3

from dataclasses import dataclass, field


@dataclass
class Config:
    model_name_or_path: str = field(metadata={"help": "path to model"})
    data_dir: str = field(metadata={"help": "path to dataset directory"})

    ## Data Config ##
    ## =========== ##

    # Data directories
    output_name: str | None = field(default=None, metadata={"help": "path to output directory"})

    # Output tags
    definition: bool = field(default=False, metadata={"help": "predict definitions"})
    theorem: bool = field(default=False, metadata={"help": "predict theorems"})
    example: bool = field(default=False, metadata={"help": "predict examples"})
    proof: bool = field(default=False, metadata={"help": "predict proofs"})
    name: bool = field(default=False, metadata={"help": "predict names"})
    reference: bool = field(default=False, metadata={"help": "predict references"})

    # Data processing
    data_context_len: int = field(default=512, metadata={"help": "context length for data"})
    data_overlap_len: int = field(default=512, metadata={"help": "overlap length for data"})
    examples_as_theorems: bool = field(default=False, metadata={"help": "convert all the examples to theorems"})
    train_only_tags: list[str] | None = field(default=None, metadata={"help": "tags to train on"})

    # KFold stuff
    k_fold: int = field(default=0, metadata={"help": "number of folds to make"})
    fold: int = field(default=0, metadata={"help": "fold index to use"})

    ## Model Config ##
    ## ============ ##

    # Model Parameters
    context_len: int = field(default=512, metadata={"help": "model max context length"})
    overlap_len: int = field(default=512, metadata={"help": "model overlap length"})
    dropout: float = field(default=0.0, metadata={"help": "model dropout"})
    crf_loss_reduction: str = field(default="token_mean", metadata={"help": "crf loss reduction type"})
    crf_segment_length: int = field(default=0, metadata={"help": "semi-crf segment length"})
    logit_pooling: str = field(default="mean", metadata={"help": "bert logit pooling strat for overlapping tokens"})

    # Initialization
    freeze_base: bool = field(default=False, metadata={"help": "freeze bert weights"})
    freeze_crf: bool = field(default=False, metadata={"help": "freeze crf weights"})
    freeze_base_after_steps: int = field(default=-1, metadata={"help": "freeze base after k training steps"})
    randomize_last_layer: bool = field(default=False, metadata={"help": "randomize last bert layer weights"})
    model_debug: bool = field(default=False, metadata={"help": "toggle debug mode"})

    # Checkpoint
    checkpoint: str | None = field(default=None, metadata={"help": "pretrained checkpoint to load"})

    # Stacked model only
    stacked: bool = field(default=False, metadata={"help": "toggle stacked model"})
    use_input_ids: bool = field(default=False, metadata={"help": "use input ids for stacked model"})

    ## Data Config ##
    ## ============ ##

    ## Training Config ##
    ## =============== ##
    run_train: bool = field(default=False, metadata={"help": "run training"})
    run_test: bool = field(default=False, metadata={"help": "run training"})
    run_predict: bool = field(default=False, metadata={"help": "run prediction"})
    run_tune: bool = field(default=False, metadata={"help": "run tuning"})

    average_checkpoints: bool = field(default=False, metadata={"help": "average last K checkpoints after training"})
    predict_on_train: bool = field(default=False, metadata={"help": "run testing on train dataset"})

    trials: int = field(default=1, metadata={"help": "number of tuning trials to run"})

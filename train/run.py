"""
Basic training script. Will load the `naver-clova-ix/cord-v2` dataset 
and train a `google/pix2struct-base` model on it.
"""

import os
import sys
import json
from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from polyleven import levenshtein
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    HfArgumentParser,
    Pix2StructProcessor,
)

from flash_pix2struct import Pix2StructForConditionalGeneration

filepath = Path(__file__).resolve().parent


class P2STrainer(Seq2SeqTrainer):
    """
    Flash attention can only be used in fp16 or bf16 mode.
    This trainer ensures that evaluation and prediction are done in the correct dtype.

    If you want to do prediction in fp32, you need to use the regular `Pix2StructForConditionalGeneration`
    from `transformers` and then use `Seq2SeqTrainer` as well.
    """

    def __init__(self, *args, eval_dtype="bf16", **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dtype = eval_dtype

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        dtype = torch.bfloat16 if self.eval_dtype == "bf16" else torch.float16
        enabled = self.eval_dtype in {"bf16", "fp16"}
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )


def preprocess(examples, processor, config):
    """
    Tokenize the json strings and encode the images.
    Pad sequences to max length in the batch, make sure
    it is a multiple of `config.pad_to_multiple_of`.

    Args:
        examples: list of examples
        processor: `Pix2StructProcessor`
        config: `Config`

    Returns:
        dict: dictionary with keys `flattened_patches`, `attention_mask`, `labels`
    """

    texts = []
    for x in examples:
        j = json.loads(x["ground_truth"])

        texts.append(json.dumps(j["gt_parse"]["menu"]))

    images = [x["image"] for x in examples]

    input_ids = processor.tokenizer(
        texts,
        max_length=config.max_seq_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids

    seq_len = len(input_ids[0])
    batch_size = input_ids.size(0)

    # Ensure the maximum length is a multiple of config.pad_to_multiple_of
    if seq_len % config.pad_to_multiple_of != 0:
        seq_len += config.pad_to_multiple_of - (seq_len % config.pad_to_multiple_of)

    # Padding if necessary
    if input_ids.size(1) < seq_len:
        padding = torch.zeros(batch_size, seq_len - input_ids.size(1), dtype=torch.long)
        input_ids = torch.cat([input_ids, padding], dim=-1)

    # Don't calculate loss for pad tokens
    input_ids[input_ids == processor.tokenizer.pad_token_id] = -100

    text_prompt = None
    if config.prompt is not None:
        text_prompt = config.prompt

    texts = [text_prompt] * len(images) if text_prompt is not None else None

    encoding = processor(
        images=images,
        text=texts,
        max_patches=config.max_patches,
        return_tensors="pt",
        font_path=str(filepath / "Arial.TTF"),
    )

    return {
        **encoding,
        "labels": input_ids,
    }


def compute_metrics(eval_predictions, tokenizer):
    """
    Compute the levenshtein distance between the predictions and the labels.
    """
    predictions, label_ids = eval_predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = predictions.argmax(-1)

    # Ignore padding tokens
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "levenshtein": np.mean(
            [
                levenshtein(decoded_preds[i], decoded_labels[i])
                / max(len(decoded_labels[i]), len(decoded_preds[i]))
                for i in range(len(decoded_preds))
            ]
        )
    }


@dataclass
class Config:
    model_name_or_path: str = field(
        default="google/pix2struct-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    max_patches: int = field(
        default=1024, metadata={"help": "Number of patches for image model"}
    )

    max_seq_length: int = field(default=512, metadata={"help": "Max length for text"})

    pad_to_multiple_of: int = field(default=16, metadata={"help": "Pad to multiple of"})

    prompt: str = field(
        default=None,
        metadata={"help": "Prompt that will be put on the image"},
    )


def main():
    parser = HfArgumentParser((Config, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        config_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        config_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    ds = load_dataset("naver-clova-ix/cord-v2")

    model = Pix2StructForConditionalGeneration.from_pretrained(
        config_args.model_name_or_path
    )
    is_vqa = config_args.prompt is not None
    processor = Pix2StructProcessor.from_pretrained(
        config_args.model_name_or_path, is_vqa=is_vqa
    )

    eval_dtype = "fp32"

    if training_args.bf16:
        eval_dtype = "bf16"
    elif training_args.fp16:
        eval_dtype = "fp16"

    trainer = P2STrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor.tokenizer,
        data_collator=partial(preprocess, processor=processor, config=config_args),
        compute_metrics=partial(compute_metrics, tokenizer=processor.tokenizer),
        eval_dtype=eval_dtype,
    )

    processor.save_pretrained(training_args.output_dir)

    # Show first example
    sample = preprocess([ds["train"][0]], processor, config_args)
    l = sample["labels"]
    l = l[l != -100]
    print(l)
    print(processor.tokenizer.decode(l.tolist()))

    print("last checkpoint:", training_args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()

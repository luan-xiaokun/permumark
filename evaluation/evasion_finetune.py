"""Watermark evasion by fine-tuning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from eval_utils import compare_watermarks
from permumark import PermutationWatermark
from permumark.watermark import PermutationWatermarkInsertionResult


torch.set_float32_matmul_precision("high")


@dataclass
class FinetuneConfig:
    """Configuration for lora fine-tuning."""

    output_dir: str = "models/finetune"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 1000
    num_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    max_length_percentile: float = 0.9


@dataclass
class WatermarkConfig:
    """Configuration for watermark."""

    max_corrupt_prob: float = 1e-4
    total_id_num: int = 10_000_000
    evaluation_points: list[int] | None = None
    column_multipliers: list[int] | None = None


class TokenSaveCallback(TrainerCallback):
    """
    Callback to save model weights when reached given number of tokens and stop training.
    :param save_token_thresholds: a list of number of tokens to save
    """

    def __init__(self, save_token_thresholds: list[int]) -> None:
        super().__init__()
        self.save_token_thresholds = list(sorted(save_token_thresholds))
        self.threshold_names = [pretty_token_num(t) for t in self.save_token_thresholds]
        self.last_save_index = 0

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """
        Save the model when the model has been trained on given number of tokens.
        After reaching the last given threshold, stop training.
        :param args: training arguments, a TrainingArguments instance
        :param state: trainer state, a TrainerState instance
        :param control: trainer control, a TrainerControl instance
        :param kwargs: other arguments
        :return: None
        """
        while (
            self.last_save_index < len(self.save_token_thresholds)
            and state.num_input_tokens_seen
            >= self.save_token_thresholds[self.last_save_index]
        ):
            threshold_name = self.threshold_names[self.last_save_index]
            save_dir = f"{args.output_dir}/checkpoint-{threshold_name}_tokens"
            kwargs["model"].save_pretrained(save_dir)
            self.last_save_index += 1
            log_print(
                f"Model saved at {save_dir} after processing {threshold_name} tokens."
            )

        if self.last_save_index >= len(self.save_token_thresholds):
            control.should_training_stop = True
            log_print(f"Reached {self.threshold_names[-1]} tokens. Stopping training.")


def log_print(*args, **kwargs) -> None:
    """
    Print with formatted datetime.
    :param args: arguments to print
    :param kwargs: kwargs to print
    :return: None
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", *args, **kwargs)


def pretty_token_num(token_num: int) -> str:
    """
    Convert number of tokens to pretty printed number, e.g., 20M, 5B.
    :param token_num: number of tokens
    :return: pretty printed number
    """
    suffixes = ["", "K", "M", "B"]
    k_power = 0
    while not (1000**k_power <= token_num < 1000 ** (k_power + 1)):
        k_power += 1
    prefix = str(token_num // 1000**k_power)
    return f"{prefix}{suffixes[k_power]}"


def get_token_lengths(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> list[int]:
    """
    Get token lengths in the dataset.
    :param dataset: dataset to tokenize
    :param tokenizer: the tokenizer
    :return: a list of token lengths
    """

    def tokenize_length(example):
        return {
            "length": len(tokenizer(example["text"], truncation=False)["input_ids"])
        }

    length_dataset = dataset.map(tokenize_length, num_proc=8)
    return length_dataset["length"]


def preprocess_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, percentile: float = 0.95
) -> Dataset:
    """
    Preprocess the dataset for fine-tuning.
    :param dataset: dateset used for fine-tuning
    :param tokenizer: tokenizer of the model
    :param percentile: only consider lengths at this percentile
    :return: a preprocessed dataset
    """
    lengths = get_token_lengths(dataset, tokenizer)
    max_length = int(torch.tensor(lengths).float().quantile(percentile))
    log_print(
        f"Max length: {max(lengths)}, {percentile:.1%} percentile length: {max_length}"
    )

    def tokenize(example):
        inputs = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    return dataset.map(tokenize, remove_columns=["text"], num_proc=8)


def finetune_model(
    model_path: str,
    dataset: Dataset,
    token_thresholds: list[int],
    finetune_config: FinetuneConfig = FinetuneConfig(),
    insert_watermark: bool = True,
    watermark_config: WatermarkConfig = WatermarkConfig(),
) -> PeftModel:
    """
    Fine-tune a pre-trained transformer model using lora.
    :param model_path: path to pre-trained model
    :param dataset: dataset for fine-tuning
    :param token_thresholds: list of number of tokens to save
    :param finetune_config: fine-tuning configuration
    :param insert_watermark: whether to insert watermark
    :param watermark_config: configuration of watermark
    :return: the fine-tuned model
    """
    log_print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenizer_exceptions=False, trust_remote_code=True
    )

    output_dir = Path(finetune_config.output_dir) / Path(*Path(model_path).parts[-2:])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_print(f"Output directory set to {output_dir}")

    # embed watermark
    if insert_watermark:
        pw = PermutationWatermark(
            model.config,
            watermark_config.max_corrupt_prob,
            watermark_config.total_id_num,
            watermark_config.evaluation_points,
            watermark_config.column_multipliers,
        )
        identity = pw.generate_random_identity()
        insert_res = pw.insert_watermark(model, identity)
        log_print(f"Inserted identity: {identity}")
        log_print(f"Inserted watermark: {insert_res.watermark}")
        watermark_config = pw.to_dict()
        watermark_config["identity"] = identity
        with open(output_dir / "watermark.json", "w") as f:
            json.dump(watermark_config, f)

    if tokenizer.pad_token is None:
        if "Llama-3" in model_path:
            tokenizer.pad_token_id = 128004
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = preprocess_dataset(
        dataset, tokenizer, finetune_config.max_length_percentile
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=finetune_config.r,
        lora_alpha=finetune_config.lora_alpha,
        lora_dropout=finetune_config.lora_dropout,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=finetune_config.learning_rate,
        weight_decay=finetune_config.weight_decay,
        warmup_ratio=finetune_config.warmup_ratio,
        warmup_steps=finetune_config.warmup_steps,
        num_train_epochs=finetune_config.num_epochs,
        per_device_train_batch_size=finetune_config.per_device_train_batch_size,
        gradient_accumulation_steps=finetune_config.gradient_accumulation_steps,
        bf16=finetune_config.bf16,
        include_num_input_tokens_seen=True,
        dataloader_num_workers=8,
        # auto_find_batch_size=True,
        save_steps=5000,
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        callbacks=[TokenSaveCallback(token_thresholds)],
    )

    trainer.train()

    return peft_model


def load_finetune_model(
    base_model_path: str,
    peft_model_path: str,
    token_num: int,
) -> tuple[
    PreTrainedModel,
    PermutationWatermark | None,
    PermutationWatermarkInsertionResult | None,
]:
    """
    Load a fine-tuned transformer model from given model path.
    :param base_model_path: path to the base pre-trained model
    :param peft_model_path: path to the fine-tuned weights
    :param token_num: number of tokens trained with, used to determine checkpoint path
    :return: a transformer model with the same architecture as the base model
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", trust_remote_code=True
    )
    watermark_path = Path(peft_model_path) / "watermark.json"
    pw = None
    insert_res = None
    if watermark_path.exists():
        watermark_config = json.loads(watermark_path.read_text())
        pw = PermutationWatermark.from_dict(watermark_config)
        insert_res = pw.insert_watermark(base_model, watermark_config["identity"])
        print(f"Inserted identity: {watermark_config['identity']}")
        print(f"Inserted watermark: {insert_res.watermark}")
    ckpt_folder = f"checkpoint-{pretty_token_num(token_num)}_tokens"
    model = PeftModel.from_pretrained(
        base_model, str(Path(peft_model_path) / ckpt_folder)
    )
    return model.merge_and_unload(), pw, insert_res


def main():
    from dataclasses import fields

    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--thresholds", "-t", nargs="+", type=int)
    parser.add_argument("--dataset_path", default="datasets/Salesforce/wikitext")
    parser.add_argument("--dataset_name", default="wikitext-103-v1")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("-r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_length_percentile", type=float)

    parser.add_argument("--max_corrupt_prob", type=float)
    parser.add_argument("--total_id_num", type=int)
    parser.add_argument("--evaluation_points", nargs="*", type=int)
    parser.add_argument("--column_multipliers", nargs="*", type=int)

    args = parser.parse_args()

    log_print("Arguments:")
    for arg, value in vars(args).items():
        if value is not None:
            print(f"- {arg}: {value}")

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split=args.dataset_split,
        cache_dir=".cache/huggingface/datasets",
    )
    finetune_config_args = {
        field.name: getattr(args, field.name)
        for field in fields(FinetuneConfig)
        if hasattr(args, field.name) and getattr(args, field.name) is not None
    }
    finetune_config = FinetuneConfig(**finetune_config_args)
    watermark_config_args = {
        field.name: getattr(args, field.name)
        for field in fields(WatermarkConfig)
        if hasattr(args, field.name) and getattr(args, field.name) is not None
    }
    watermark_config = WatermarkConfig(**watermark_config_args)

    finetune_model(
        args.model_path,
        dataset,
        args.thresholds,
        finetune_config=finetune_config,
        insert_watermark=True,
        watermark_config=watermark_config,
    )


def test_watermark():
    model_path = "../models/meta-llama/Llama-3.1-8B"
    peft_model_path = "../models/finetune/meta-llama/Llama-3.1-8B"
    token_num = 50000000
    model, pw, insert_res = load_finetune_model(model_path, peft_model_path, token_num)
    source = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=True
    )
    extract_res = pw.extract_watermark(source, model)
    print(f"Extracted watermark: {extract_res.watermark}")
    print(f"Extracted identity: {extract_res.identity}")


def evaluate_finetune_robustness(
    model_path: str, source: PreTrainedModel, finetune_weights_dir: str
):
    peft_path = Path(finetune_weights_dir) / Path(*Path(model_path).parts[-2:])
    token_nums = [i * 1_000_000 for i in [1, 5, 10, 50, 100]]
    for token_num in token_nums:
        model, pw, insert_res = load_finetune_model(
            model_path, str(peft_path), token_num
        )
        extract_res = pw.extract_watermark(source, model)
        diff, total, robustness = compare_watermarks(insert_res, extract_res)
        print(
            f"Fine-tuned ({pretty_token_num(token_num)} tokens): "
            f"{diff}/{total} corrupted digits\n"
            f"Robustness: {robustness}"
        )


if __name__ == "__main__":
    test_watermark()
    # main()

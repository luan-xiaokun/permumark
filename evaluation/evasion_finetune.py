"""Watermark evasion by fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    PreTrainedTokenizer,
)


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
    num_epochs: int = 10
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    max_length_percentile: float = 0.95


class TokenSaveCallback(TrainerCallback):
    """
    Callback to save model weights when reached given number of tokens and stop training.
    :param save_token_thresholds: a list of number of tokens to save
    """

    def __init__(self, save_token_thresholds: list[float]) -> None:
        super().__init__()
        self.save_token_thresholds = list(sorted(int(t) for t in save_token_thresholds))
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
        print(
            f"Calling on_step_end of TokenSaveCallback, seen token num: {state.num_input_tokens_seen}"
        )
        while (
            self.last_save_index < len(self.save_token_thresholds)
            and state.num_input_tokens_seen
            >= self.save_token_thresholds[self.last_save_index]
        ):
            threshold_name = self.threshold_names[self.last_save_index]
            save_dir = f"{args.output_dir}/checkpoint-{threshold_name}_tokens"
            kwargs["model"].save_pretrained(save_dir)
            self.last_save_index += 1
            print(
                f"Model saved at {save_dir} after processing {threshold_name} tokens."
            )

        if self.last_save_index >= len(self.save_token_thresholds):
            control.should_training_stop = True
            print(f"Reached {self.threshold_names[-1]} tokens. Stopping training.")


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

    length_dataset = dataset.map(tokenize_length)
    return length_dataset["length"]


def preprocess_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, percentile: float = 0.95
) -> Dataset:
    lengths = get_token_lengths(dataset, tokenizer)
    max_length = int(torch.tensor(lengths).float().quantile(percentile))
    print(max(lengths), max_length)

    def tokenize(example):
        inputs = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    return dataset.map(tokenize, remove_columns=["text"])


def finetune_model(
    model_path: str,
    dataset: Dataset,
    token_thresholds: list[float],
    config: FinetuneConfig = FinetuneConfig(),
) -> PeftModel:
    """
    Fine-tune a pre-trained transformer model using lora.
    :param model_path: path to pre-trained model
    :param dataset: dataset for fine-tuning
    :param token_thresholds: list of number of tokens to save
    :param config: fine-tuning configuration
    :return: the fine-tuned model
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenizer_exceptions=False, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        if "Llama-3" in model_path:
            tokenizer.pad_token_id = 128004
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = preprocess_dataset(
        dataset, tokenizer, config.max_length_percentile
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    output_dir = Path(config.output_dir) / Path(*Path(model_path).parts[-2:])
    print(output_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.bf16,
        include_num_input_tokens_seen=True,
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        callbacks=[TokenSaveCallback(save_token_thresholds=token_thresholds)],
    )

    trainer.train()

    return peft_model


def load_finetune_model(base_model_path: str, peft_model_path: str) -> PreTrainedModel:
    """
    Load a fine-tuned transformer model from given model path.
    :param base_model_path: path to the base pre-trained model
    :param peft_model_path: path to the fine-tuned weights
    :return: a transformer model with the same architecture as the base model
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    return model.merge_and_unload()


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--thresholds", "-t", nargs="+", type=float)
    parser.add_argument("--dataset_path", default="datasets/Salesforce/wikitext")
    parser.add_argument("--dataset_name", default="wikitext-2-v1")
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

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset_path, args.dataset_name, split=args.dataset_split
    )
    config_args = {
        field.name: getattr(args, field.name)
        for field in fields(FinetuneConfig)
        if hasattr(args, field.name) and getattr(args, field.name) is not None
    }
    config = FinetuneConfig(**config_args)

    finetune_model(args.model_path, dataset, args.thresholds, config)


if __name__ == "__main__":
    main()

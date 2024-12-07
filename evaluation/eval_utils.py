"""Utility for transformer model modification and evaluation."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.quantization
import tqdm
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import processors
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def quantize_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibrate_dataset: Dataset,
    num_samples: int | None = None,
) -> None:
    """
    Quantize the model using post-training quantization.
    :param model: model to quantize
    :param tokenizer: tokenizer for the model
    :param calibrate_dataset: dataset used for calibration
    :param num_samples: Defaults to None, number of samples to use for calibration
    :return None
    """

    def calibrate_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        num_samples: int | None = None,
    ):
        model.eval()
        model.to(device)
        if num_samples is None:
            num_samples = len(dataset)

        with torch.no_grad():
            for i, sample in tqdm.tqdm(enumerate(dataset), total=num_samples):
                if i >= num_samples:
                    break
                inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True)

                model(**inputs.to(device))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    copy = deepcopy(model)
    copy.eval()
    copy.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    copy.model.embed_tokens.qconfig = None
    torch.quantization.prepare(copy, inplace=True)

    calibrate_model(copy, tokenizer, calibrate_dataset, num_samples=num_samples)
    copy.to("cpu")
    torch.quantization.convert(copy, inplace=True)

    for layer, q_layer in zip(model.model.layers, copy.model.layers):
        layer.self_attn.q_proj.weight.data = (
            q_layer.self_attn.q_proj.weight().dequantize()
        )
        layer.self_attn.k_proj.weight.data = (
            q_layer.self_attn.k_proj.weight().dequantize()
        )
        layer.self_attn.v_proj.weight.data = (
            q_layer.self_attn.v_proj.weight().dequantize()
        )
        layer.self_attn.o_proj.weight.data = (
            q_layer.self_attn.o_proj.weight().dequantize()
        )
        layer.mlp.gate_proj.weight.data = q_layer.mlp.gate_proj.weight().dequantize()
        layer.mlp.up_proj.weight.data = q_layer.mlp.up_proj.weight().dequantize()
        layer.mlp.down_proj.weight.data = q_layer.mlp.down_proj.weight().dequantize()
    model.lm_head.weight.data = copy.lm_head.weight().dequantize()

    del copy


def prune_model(model, amount=0.5) -> None:
    """
    Prune the model using global unstructured pruning.
    :param model: model to prune
    :param amount: ratio of pruned parameters. Defaults to 0.5
    :return None
    """
    parameters_to_prune = []
    for layer in model.model.layers:
        parameters_to_prune.extend(
            (
                (layer.self_attn.q_proj, "weight"),
                (layer.self_attn.k_proj, "weight"),
                (layer.self_attn.v_proj, "weight"),
                (layer.self_attn.o_proj, "weight"),
                (layer.mlp.gate_proj, "weight"),
                (layer.mlp.up_proj, "weight"),
                (layer.mlp.down_proj, "weight"),
            )
        )
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.l1_unstructured, amount=amount
    )
    for layer in model.model.layers:
        prune.remove(layer.self_attn.q_proj, "weight")
        prune.remove(layer.self_attn.k_proj, "weight")
        prune.remove(layer.self_attn.v_proj, "weight")
        prune.remove(layer.self_attn.o_proj, "weight")
        prune.remove(layer.mlp.gate_proj, "weight")
        prune.remove(layer.mlp.up_proj, "weight")
        prune.remove(layer.mlp.down_proj, "weight")


def add_noise_to_model(model, std: float = 1.0) -> None:
    """
    Add Gaussian noise to the model's parameters.
    :param model: model to add noise to
    :param std: standard deviation of the Gaussian noise. Defaults to 1.0
    :return None
    """
    with torch.no_grad():
        model.model.embed_tokens.weight.data += std * torch.randn_like(
            model.model.embed_tokens.weight.data
        )
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.data += std * torch.randn_like(
                layer.self_attn.q_proj.weight.data
            )
            layer.self_attn.k_proj.weight.data += std * torch.randn_like(
                layer.self_attn.k_proj.weight.data
            )
            layer.self_attn.v_proj.weight.data += std * torch.randn_like(
                layer.self_attn.v_proj.weight.data
            )
            layer.self_attn.o_proj.weight.data += std * torch.randn_like(
                layer.self_attn.o_proj.weight.data
            )
            layer.mlp.gate_proj.weight.data += std * torch.randn_like(
                layer.mlp.gate_proj.weight.data
            )
            layer.mlp.up_proj.weight.data += std * torch.randn_like(
                layer.mlp.up_proj.weight.data
            )
            layer.mlp.down_proj.weight.data += std * torch.randn_like(
                layer.mlp.down_proj.weight.data
            )
        model.lm_head.weight.data += std * torch.randn_like(model.lm_head.weight.data)


def lora_finetune_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: DatasetDict
) -> PreTrainedModel:
    """
    Fine-tune the model using lora.
    :param model: model to fine-tune
    :param tokenizer: tokenizer for the model
    :param dataset: fine-tuning dataset
    :return fine-tuned model
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = setup_llama3_tokenizer(tokenizer)

    def tokenize(examples):
        tokens = tokenizer(examples["text"], return_tensors="pt", padding="longest")
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=f"./models/lora_finetune/{model.config._name_or_path}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        bf16=True,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        warmup_steps=1000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    return model.merge_and_unload().cpu()


def eval_predicted_tokens(model, dataloader: DataLoader) -> list[int]:
    """
    Evaluate the model by predicting the next token in the dataset.
    :param model: model to evaluate
    :param dataloader: dataloader for evaluation
    :return predicted tokens
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    predicted_next_token_ids = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            predicted_token_id = outputs.logits[:, -1].argmax(-1).tolist()
            predicted_next_token_ids.extend(predicted_token_id)

    model.to("cpu")

    return predicted_next_token_ids


def eval_perplexity(model, tokenizer, dataset: Dataset, batch_size: int = 128) -> float:
    """
    Evaluate the model by calculating the perplexity.
    :param model: model to evaluate
    :param tokenizer: tokenizer for the model
    :param dataset: dataset to evaluate
    :param batch_size: batch size for evaluation
    :return perplexity
    """
    texts = dataset["text"]
    input_ids_list = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        size = min(batch_size, len(texts) - i)
        encodings = tokenizer(texts[i : i + size], return_tensors="pt", padding=True)
        for ids in encodings["input_ids"]:
            input_ids_list.extend(ids[ids != tokenizer.pad_token_id].tolist())

    stride = 256
    seq_len = len(input_ids_list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlls = []

    model.eval()
    model.to(device)

    for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + 2 * stride, seq_len)
        input_ids = (
            torch.tensor(input_ids_list[begin_loc:end_loc]).unsqueeze(0).to(device)
        )
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


def setup_llama3_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Set up the tokenizer for the Llama model.
    :param tokenizer: tokenizer of the llama3 model
    :return tokenizer with bos, eos and padding properly set
    """
    # using <|finetune_right_pad_id|> token for padding, from:
    # https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418
    tokenizer.pad_token_id = 128004
    # modify tokenizer to add eos at the end of the text, from:
    # https://github.com/huggingface/transformers/issues/30947#issuecomment-2128057992
    bos = "<|begin_of_text|>"
    eos = "<|end_of_text|>"
    tokenizer._tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[
                    (bos, tokenizer.bos_token_id),
                    (eos, tokenizer.eos_token_id),
                ],
            ),
        ]
    )
    return tokenizer

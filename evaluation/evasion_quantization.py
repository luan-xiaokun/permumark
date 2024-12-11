"""Watermark evasion by quantization."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.ao.quantization
import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from optimum.gptq import GPTQQuantizer
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from eval_utils import compare_watermarks
from evasion_finetune import preprocess_dataset
from permumark import PermutationWatermark


def static_post_training_quantization(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, calibrate_dataset: Dataset
) -> PreTrainedModel:
    """
    Static post-training quantization (PTQ).
    :param model: transformer model to quantize
    :param tokenizer: tokenizer of the model
    :param calibrate_dataset: dataset for calibration
    :return: quantized model
    """
    # prepare
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig("qconfig")
    model.model.embed_tokens.qconfig = None
    torch.ao.quantization.prepare(model, inplace=True)
    # calibration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = preprocess_dataset(calibrate_dataset, tokenizer, 1.0)
    dataloader = DataLoader(dataset, batch_size=16)
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            model(**batch.to(device))
    # quantization
    model.to("cpu")
    torch.quantization.convert(model, inplace=True)
    return model


def static_post_training_dequantization(model: PreTrainedModel) -> PreTrainedModel:
    """
    De-quantize weights in the model.
    :param model: a quantized model
    :return: de-quantized model
    """
    for layer in model.model.layers:
        layer.self_attn.q_proj.weight.data = (
            layer.self_attn.q_proj.weight.data.dequantize()
        )
        layer.self_attn.k_proj.weight.data = (
            layer.self_attn.k_proj.weight.data.dequantize()
        )
        layer.self_attn.v_proj.weight.data = (
            layer.self_attn.v_proj.weight.data.dequantize()
        )
        layer.self_attn.o_proj.weight.data = (
            layer.self_attn.o_proj.weight.data.dequantize()
        )
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data.dequantize()
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data.dequantize()
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data.dequantize()
    model.lm_head.weight.data = model.lm_head.weight.data.dequantize()
    return model


def gptq_quantization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibrate_dataset: Dataset,
):
    quantizer = GPTQQuantizer(bits=4, dataset=calibrate_dataset["text"])
    model = quantizer.quantize_model(model, tokenizer)
    return model


def evaluate_quantization_robustness(
    model_path: str,
    source: PreTrainedModel,
    bits: int,
    dataset: Dataset,
):
    assert bits in (4, 8), f"Only support 4 bits and 8 bits, got {bits} bits."

    model = deepcopy(source)
    pw = PermutationWatermark(model.config)
    identity = pw.generate_random_identity()
    insert_res = pw.insert_watermark(model, identity)
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
    if bits == 8:
        quantized_model = static_post_training_quantization(model, tokenizer, dataset)
        recovered_model = static_post_training_dequantization(quantized_model)
    else:
        quantized_model = gptq_quantization(model, tokenizer, dataset)
        # TODO implement dequantization for GPTQ
        recovered_model = quantized_model
    extract_res = pw.extract_watermark(source, recovered_model)
    diff, total, robustness = compare_watermarks(insert_res, extract_res)
    print(
        f"Quantization ({bits} bits): {diff}/{total} corrupted digits\n"
        f"Robustness: {robustness}"
    )

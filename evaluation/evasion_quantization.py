"""Watermark evasion by quantization."""

from __future__ import annotations

import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.ao.quantization
import tqdm
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from eval_utils import compare_watermarks, preprocess_dataset
from permumark import PermutationWatermark


auto_gptq_loggers = [
    "auto_gptq.modeling._base",
    "auto_gptq.modeling._utils",
    "auto_gptq.quantization.config",
]
for logger_name in auto_gptq_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


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
    model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    model.model.embed_tokens.qconfig = None
    torch.ao.quantization.prepare(model, inplace=True)
    # calibration
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"  # NOTE currently our GPU does not have enough CUDA memory
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = preprocess_dataset(calibrate_dataset, tokenizer, 1.0)
    dataloader = DataLoader(
        dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer)
    )
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            model(**batch.to(device))
    # quantization
    model.to("cpu")
    torch.quantization.convert(model, inplace=True)
    return model


def dequantize_model(
    model: nn.Module,
    copy: PreTrainedModel,
    bits: int,
) -> PreTrainedModel:
    """
    De-quantize weights in the model.
    :param model: a quantized model
    :param copy: a copy of the original model
    :param bits: number of bits to quantize
    :return: de-quantized model
    """
    assert bits in (2, 3, 4, 8), f"Unsupported bits: {bits}"

    def static_ptq_dequantize_weight(q_module: nn.Module, module: nn.Module):
        dequantized = q_module.weight().dequantize()
        module.weight.data = dequantized

    def gptq_dequantize_weight(q_linear: QuantLinear, linear: nn.Module):
        # reference
        # https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py
        if q_linear.bits in (2, 4, 8):
            wf = torch.tensor(
                list(range(0, 32, q_linear.bits)), dtype=torch.int32
            ).unsqueeze(0)
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(q_linear.qzeros, 2).expand(-1, -1, 32 // q_linear.bits),
                wf.unsqueeze(0),
            ).to(torch.int8)
            zeros = zeros + 1
            zeros = torch.bitwise_and(zeros, 2**q_linear.bits - 1)
            zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
            scales = q_linear.scales
            scales = scales.reshape(-1, 1, scales.shape[-1])
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(q_linear.qweight, 1).expand(
                    -1, 32 // q_linear.bits, -1
                ),
                wf.unsqueeze(-1),
            ).to(torch.int8)
            weight = torch.bitwise_and(weight, 2**q_linear.bits - 1)
        else:
            wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)
            zeros = q_linear.qzeros.reshape(
                q_linear.qzeros.shape[0], q_linear.qzeros.shape[1] // 3, 3, 1
            ).expand(-1, -1, -1, 12)
            zeros = zeros >> wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
                (zeros[:, :, 1, 0] << 2) & 0x4
            )
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
                (zeros[:, :, 2, 0] << 1) & 0x6
            )
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2
            )
            zeros = zeros + 1
            zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
            scales = q_linear.scales
            scales = scales.reshape(-1, 1, scales.shape[-1])
            weight = q_linear.qweight.reshape(
                q_linear.qweight.shape[0] // 3, 3, 1, q_linear.qweight.shape[1]
            ).expand(-1, -1, 12, -1)
            weight = (weight >> wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat(
                [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
            )

        weight = weight.reshape(-1, q_linear.group_size, weight.shape[2])
        weight = scales * (weight - zeros)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).t()
        weight = weight.contiguous()
        linear.weight.data = weight

    if bits == 8:
        dequantize_weight = static_ptq_dequantize_weight
    else:
        dequantize_weight = gptq_dequantize_weight

    for q_layer, layer in zip(model.model.layers, copy.model.layers):
        dequantize_weight(q_layer.self_attn.q_proj, layer.self_attn.q_proj)
        dequantize_weight(q_layer.self_attn.k_proj, layer.self_attn.k_proj)
        dequantize_weight(q_layer.self_attn.v_proj, layer.self_attn.v_proj)
        dequantize_weight(q_layer.self_attn.o_proj, layer.self_attn.o_proj)
        dequantize_weight(q_layer.mlp.gate_proj, layer.mlp.gate_proj)
        dequantize_weight(q_layer.mlp.up_proj, layer.mlp.up_proj)
        dequantize_weight(q_layer.mlp.down_proj, layer.mlp.down_proj)
    if bits == 8:
        dequantize_weight(model.lm_head, copy.lm_head)
    del model

    return copy


def evaluate_quantization_robustness(
    model_path: str,
    source: PreTrainedModel,
    pw: PermutationWatermark,
    bits: int,
    dataset: Dataset,
    verbose: bool = False,
):
    """
    Apply naive static PTQ or GPTQ for quantization, and evaluate the robustness
    of the watermark under quantization modification.
    :param model_path: path to the original model
    :param source: source model without watermark
    :param pw: a PermutationWatermark instance
    :param bits: quantization bits
    :param dataset: dataset for quantization
    :param verbose: verbose output
    :return: None
    """
    assert bits in (2, 3, 4, 8), f"Only support 2,3,4,8 bits, got {bits} bits."
    print(f"Evaluating quantization robustness ({bits}-bits)")
    identity = pw.generate_random_identity()
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
    if bits == 8:
        model = deepcopy(source)
        insert_res = pw.insert_watermark(model, identity, verbose=verbose)
        copy_model = deepcopy(model).to(torch.float16)
        quantized_model = static_post_training_quantization(model, tokenizer, dataset)
    else:
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config=BaseQuantizeConfig(
                bits=bits, group_size=128, desc_act=False
            ),
        )
        insert_res = pw.insert_watermark(model.model, identity, verbose=verbose)
        copy_model = deepcopy(model.model).to(torch.float16)
        if tokenizer.pad_token is None:
            if "Llama-3" in model_path:
                tokenizer.pad_token_id = 128004
            tokenizer.pad_token = tokenizer.eos_token
        model.to("cuda")  # auto-gptq requires quantization on cuda
        model.quantize(list(preprocess_dataset(dataset, tokenizer, 1.0)))
        quantized_model = model.model

    quantized_model.to("cpu")
    recovered_model = dequantize_model(quantized_model, copy_model, bits)
    extract_res = pw.extract_watermark(source, recovered_model, verbose=verbose)
    diff, total, robustness = compare_watermarks(insert_res, extract_res)
    print(
        f"Quantization ({bits} bits): {diff}/{total} corrupted digits\n"
        f"Robustness: {robustness}"
    )

"""Evaluation for baseline method."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

import eval_utils
from permumark.watermark import FunctionInvariantTransformationWatermark

DATASET = {
    "path": "datasets/Salesforce/wikitext",
    "name": "wikitext-2-v1",
    "split": "train",
}
QUANTIZATION_DATASET_SIZE = 512
PRUNING_CALIBRATION_SIZE = 32
PRUNING_TOKENIZE_MAX_LENGTH = 100


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["utility", "robustness", "efficiency"])
    parser.add_argument("model_type", choices=["llama", "gemma"])
    parser.add_argument(
        "--mode",
        choices=["permutation", "scaling", "qk-products", "all"],
        default="all",
    )
    parser.add_argument("--modification", type=str)
    parser.add_argument("--pruning_method", type=str, default="l1-unstructured")
    parser.add_argument("--pruning_amount", type=float, default=0.5)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--perm_budget", type=int, default=20)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def get_model_path(model_type: str) -> str:
    if model_type.lower() == "llama":
        return "models/meta-llama/Llama-2-7b-hf"
    if model_type.lower() == "gemma":
        return "models/google/gemma-7b"
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_utility(
    model_path: str,
    model: PreTrainedModel,
    fit: FunctionInvariantTransformationWatermark,
) -> None:
    """
    Evaluate utility of watermarked model, evaluate distortion on predicted tokens
    and change of perplexity.
    :param model_path: path to model, used to load tokenizer
    :param model: transformer model to evaluate
    :param fit: a FunctionInvariantTransformationWatermark instance
    :return: None
    """
    dataset = load_dataset(**DATASET)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # get predicted tokens and perplexity
    predicted0, ppl0 = eval_utils.eval_predicted_tokens_and_perplexity(
        model, model_path, dataset
    )
    # insert watermark and re-evaluate predicted tokens and perplexity
    fit.insert_watermark(model, fit.generate_random_identity())
    predicted1, ppl1 = eval_utils.eval_predicted_tokens_and_perplexity(
        model, model_path, dataset
    )
    diff = sum(p0 != p1 for p0, p1 in zip(predicted0, predicted1))
    total = len(predicted0)

    print(f"Token distortion: {diff}/{total} ({diff / total:.2%})")
    print(f"Perplexity {ppl0:.3f} -> {ppl1:.3f}")


def evaluate_efficiency(
    model: PreTrainedModel, fit: FunctionInvariantTransformationWatermark, repeat: int
):
    """
    Evaluate efficiency of FIT watermark, report average insertion and extraction time.
    :param model: transformer model to evaluate with
    :param fit: a PermutationWatermark instance
    :param repeat: number of repetitions
    :return: None
    """
    insert_time_list = []
    extract_time_list = []
    source = deepcopy(model)
    for _ in tqdm.tqdm(range(repeat), desc="Evaluating efficiency"):
        identity = fit.generate_random_identity()
        insert_start_time = time.time()
        fit.insert_watermark(model, identity)
        print(f"Inserted watermark: {identity}")
        insert_end_time = time.time()
        extracted_identity = fit.extract_watermark(source, model)
        print(f"Extracted watermark: {extracted_identity}")
        extract_end_time = time.time()
        if identity != extracted_identity:
            print("WARNING: extracted identity does not match inserted identity")

        insert_time_list.append(insert_end_time - insert_start_time)
        extract_time_list.append(extract_end_time - insert_end_time)

    avg_insert_time = sum(insert_time_list) / len(insert_time_list)
    avg_extract_time = sum(extract_time_list) / len(extract_time_list)
    print(f"Average insertion time: {avg_insert_time:.3f}s")
    print(f"Average extraction time: {avg_extract_time:.3f}s")


def main():
    """Baseline evaluation entrance."""
    args = get_parser().parse_args()
    model_path = get_model_path(args.model_type)
    if args.verbose:
        print(f"Task: {args.task}")

    # load model
    if args.verbose:
        print(f"Loading model {model_path}")
    torch_dtype = "auto"
    if args.modification == "quantization":
        torch_dtype = torch.float32

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )

    fit = FunctionInvariantTransformationWatermark(model.config, args.mode)
    if args.verbose:
        print(fit)

    if args.task == "utility":
        evaluate_utility(model_path, model, fit)
    elif args.task == "robustness":
        pass
    elif args.task == "efficiency":
        evaluate_efficiency(model, fit, repeat=args.repeat)


if __name__ == "__main__":
    main()

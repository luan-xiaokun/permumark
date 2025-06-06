"""Evaluation for PermuMark."""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer

import attacks
import eval_utils
from evasion_finetune import evaluate_finetune_robustness
from evasion_pruning import evaluate_pruning_robustness
from permumark import PermutationWatermark

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # the following code is used to suppress the log messages from transformers
# import logging
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for name in logging.root.manager.loggerDict:
#     logger = logging.getLogger(name)
#     if "transformers" in logger.name.lower():
#         logger.setLevel(logging.ERROR)


SUPPORTED_MODELS = ["gemma", "llama", "mistral", "qwen", "qwen2", "stablelm"]
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
    parser.add_argument(
        "task", type=str, choices=["utility", "robustness", "efficiency", "security"]
    )
    parser.add_argument("model_type", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--inv_attack", action="store_true")
    parser.add_argument("--max_corrupt_prob", type=float, default=1e-4)
    parser.add_argument("--modification", type=str)
    parser.add_argument("--perm_budget", type=int, default=20)
    parser.add_argument("--perm_type", type=str, default="random")
    parser.add_argument("--pruning_method", type=str, default="l1-unstructured")
    parser.add_argument("--pruning_amount", type=float, default=0.5)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--scale_attack", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--size", type=str, default="7b")
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--total_id_num", type=int, default=10_000_000)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def get_model_path(model_type: str, size: str) -> str:
    """
    Return the path to a model with specified model type and size.
    :param model_type: one of SUPPORTED_MODELS
    :param size: model size
    :return: path to the transformer model
    """
    if model_type.lower() == "gemma":
        return "models/google/gemma-7b"
    if model_type.lower() == "mistral":
        return "models/mistralai/Ministral-8B-Instruct-2410"
    if model_type.lower() == "stablelm":
        return "models/stabilityai/stablelm-2-12b"

    if model_type.lower() == "llama":
        if size.lower() == "1b":
            return "models/meta-llama/Llama-3.2-1B"
        if size.lower() == "3b":
            return "models/meta-llama/Llama-3.2-3B"
        if size.lower() == "7b":
            return "models/meta-llama/Llama-2-7b-hf"
        if size.lower() == "8b":
            return "models/meta-llama/Llama-3.1-8B"
        if size.lower() == "13b":
            return "models/meta-llama/Llama-2-13b-hf"

    if model_type.lower() == "qwen":
        if size.lower() == "4b":
            return "models/Qwen/Qwen1.5-4B"
        if size.lower() == "7b":
            return "models/Qwen/Qwen1.5-7B"
        if size.lower() == "14b":
            return "models/Qwen/Qwen1.5-14B"

    if model_type.lower() == "qwen2":
        if size.lower() == "7b":
            return "models/Qwen/Qwen2.5-7B"
        if size.lower() == "14b":
            return "models/Qwen/Qwen2.5-14B"

    raise ValueError(f"{model_type} with size {size} is not supported")


def evaluate_utility(
    model_path: str,
    model: PreTrainedModel,
    pw: PermutationWatermark,
) -> None:
    """
    Evaluate utility of watermarked model, evaluate distortion on predicted tokens
    and change of perplexity.
    :param model_path: path to model, used to load tokenizer
    :param model: transformer model to evaluate
    :param pw: a PermutationWatermark instance
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
    _ = pw.insert_watermark(model, pw.generate_random_identity())
    predicted1, ppl1 = eval_utils.eval_predicted_tokens_and_perplexity(
        model, model_path, dataset
    )
    diff = sum(p0 != p1 for p0, p1 in zip(predicted0, predicted1))
    total = len(predicted0)

    print(f"Token distortion: {diff}/{total} ({diff / total:.2%})")
    print(f"Perplexity {ppl0:.3f} -> {ppl1:.3f}")


def evaluate_robustness(
    model_path: str,
    model: PreTrainedModel,
    pw: PermutationWatermark,
    modification: str,
    finetune_weights_dir: str,
    pruning_method: str,
    pruning_amount: float,
    quant_bits: int,
    verbose: bool = False,
):
    """
    Evaluate robustness of PermuMark, verify integrity of watermarks under model
    modifications.
    :param model_path: path of the transformer model to insert watermark
    :param model: original model without watermark
    :param pw: a PermutationWatermark instance
    :param modification: model modification method
    :param finetune_weights_dir: directory of fine-tuning weights
    :param pruning_method: pruning method to apply
    :param pruning_amount: amount of pruned parameters
    :param quant_bits: quantization bits
    :param verbose: verbose output of watermark insertion and extraction
    :return: None
    """
    if modification == "finetune":
        evaluate_finetune_robustness(model_path, model, finetune_weights_dir, verbose)
    elif modification == "pruning":
        dataset = load_dataset(**DATASET)
        evaluate_pruning_robustness(
            model_path,
            model,
            pw,
            pruning_method,
            dataset=dataset,
            pruning_amount=pruning_amount,
            calibration_sample_num=PRUNING_CALIBRATION_SIZE,
            max_length=PRUNING_TOKENIZE_MAX_LENGTH,
            verbose=verbose,
        )
    elif modification == "quantization":
        from evasion_quantization import evaluate_quantization_robustness

        dataset = load_dataset(**DATASET).select(range(QUANTIZATION_DATASET_SIZE))
        evaluate_quantization_robustness(
            model_path, model, pw, quant_bits, dataset, verbose
        )


def evaluate_efficiency(
    model: PreTrainedModel, pw: PermutationWatermark, repeat: int
) -> None:
    """
    Evaluate efficiency of PermMark, compute average insertion time, extraction time,
    and linear assignment solving time.
    :param model: transformer model to evaluate with
    :param pw: a PermutationWatermark instance
    :param repeat: number of repetitions
    :return: None
    """
    insert_time_list = []
    extract_time_list = []
    solving_time_list = []
    source = deepcopy(model)
    for _ in tqdm(range(repeat), desc="Evaluating efficiency", total=repeat):
        identity = pw.generate_random_identity()
        insert_start_time = time.time()
        insert_res = pw.insert_watermark(model, identity)
        print(f"Inserted watermark: {insert_res.watermark}")
        insert_end_time = time.time()
        extract_res = pw.extract_watermark(source, model)
        extract_end_time = time.time()
        print(f"Extracted watermark: {extract_res.watermark}")
        assert identity == extract_res.identity

        insert_time_list.append(insert_end_time - insert_start_time)
        extract_time_list.append(extract_end_time - insert_end_time)
        solving_time_list.append(sum(extract_res.time_list))

        model = deepcopy(source)

    avg_insert_time = sum(insert_time_list) / len(insert_time_list)
    avg_extract_time = sum(extract_time_list) / len(extract_time_list)
    avg_solving_time = sum(solving_time_list) / len(solving_time_list)

    print(f"Average insertion time: {avg_insert_time:.3f}s")
    print(f"Average extraction time: {avg_extract_time:.3f}s")
    print(f"Average solving time: {avg_solving_time:.3f}s")


def evaluate_security(
    model: PreTrainedModel,
    pw: PermutationWatermark,
    perm_budget: int,
    perm_type: str,
    scale_attack: bool,
    inv_attack: bool,
    repeat: int,
    simulate: bool,
) -> None:
    """
    Evaluate security of PermuMark, apply attacks on watermarked models and check if
    the identity can still be recovered.
    :param model: transformer model to evaluate with
    :param pw: a PermutationWatermark instance
    :param perm_budget: number of corruptions on permutations to apply
    :param perm_type: type of permutations to apply
    :param scale_attack: whether apply scaling attacks
    :param inv_attack: whether apply query-key invertible matrices attacks
    :param repeat: number of repetitions for simulation
    :param simulate: use simulation for more efficient evaluation
    :return: None
    """
    atk_msg = f"Attack setting: perm_budget={perm_budget}, perm_type={perm_type}"
    if inv_attack:
        atk_msg += ", inv_attack=True"
    if scale_attack:
        atk_msg += ", scale_attack=True"
    print(atk_msg)

    if simulate:
        attack_success_count = 0
        total_erasure_num = 0
        total_corrupt_num = 0
        for _ in tqdm(range(repeat), desc="Simulating attacks", total=repeat):
            attack_success, erasure_num, corrupt_num = attacks.simulate_attacks(
                pw, perm_budget, perm_type
            )
            attack_success_count += attack_success
            total_erasure_num += erasure_num
            total_corrupt_num += corrupt_num

        print(f"Attack success rate: {attack_success_count / repeat * 100:.2f}%")
        print(f"Average number of erasures: {total_erasure_num / repeat:.2f}")
        print(f"Total number of undetected corruptions: {total_corrupt_num}")
        return

    source = deepcopy(model)
    identity = pw.generate_random_identity()
    insert_res = pw.insert_watermark(model, identity)
    attacks.attack_watermarked_model(
        model, pw, perm_budget, inv_attack, scale_attack, perm_type
    )
    extract_res = pw.extract_watermark(source, model)

    undetected_corruption_num = sum(
        i != j for i, j in zip(insert_res.watermark, extract_res.watermark)
    ) - len(extract_res.erasure_idx)
    print(f"Attack success: {extract_res.identity != identity}")
    print(f"Number of erasures: {len(extract_res.erasure_idx)}")
    print(f"Total number of undetected corruptions: {undetected_corruption_num}")


def main():
    """Evaluation entrance."""
    args = get_parser().parse_args()
    model_path = get_model_path(args.model_type, args.size)
    if args.verbose:
        print(f"Task: {args.task}")

    # load model
    if args.verbose:
        print(f"Loading model {model_path}")
    torch_dtype = args.torch_dtype
    if args.modification == "quantization":
        torch_dtype = torch.float32
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )

    # setup watermark
    pw = PermutationWatermark(model.config, args.max_corrupt_prob, args.total_id_num)
    if args.verbose:
        print(pw)

    if args.task == "utility":
        evaluate_utility(model_path, model, pw)
    elif args.task == "robustness":
        evaluate_robustness(
            model_path,
            model,
            pw,
            args.modification,
            "models/finetune",
            args.pruning_method,
            args.pruning_amount,
            args.quant_bits,
            args.verbose,
        )
    elif args.task == "efficiency":
        evaluate_efficiency(model, pw, repeat=args.repeat)
    elif args.task == "security":
        evaluate_security(
            model,
            pw,
            perm_budget=args.perm_budget,
            perm_type=args.perm_type,
            scale_attack=args.scale_attack,
            inv_attack=args.inv_attack,
            repeat=args.repeat,
            simulate=args.simulate,
        )


if __name__ == "__main__":
    main()

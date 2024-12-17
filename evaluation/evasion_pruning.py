"""Watermark evasion by pruning."""

from __future__ import annotations

from copy import deepcopy

from datasets import Dataset
from torch.nn.utils import prune
from transformers import PreTrainedModel, AutoTokenizer

from eval_utils import compare_watermarks
from permumark import PermutationWatermark
from sparse_gpt import sparse_gpt_unstructured_pruning


def l1_unstructured_prune_model(model: PreTrainedModel, amount: float = 0.5) -> None:
    """
    Prune the model using global L1 unstructured pruning.
    Note that the model is pruned inplace.
    :param model: a transformer model to prune
    :param amount: the amount of global unstructured pruning
    :return: None
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
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    for layer in model.model.layers:
        prune.remove(layer.self_attn.q_proj, "weight")
        prune.remove(layer.self_attn.k_proj, "weight")
        prune.remove(layer.self_attn.v_proj, "weight")
        prune.remove(layer.self_attn.o_proj, "weight")
        prune.remove(layer.mlp.gate_proj, "weight")
        prune.remove(layer.mlp.up_proj, "weight")
        prune.remove(layer.mlp.down_proj, "weight")


def evaluate_pruning_robustness(
    model_path: str,
    source: PreTrainedModel,
    pw: PermutationWatermark,
    pruning_method: str,
    dataset: Dataset | None = None,
    pruning_amount: float = 0.5,
    verbose: bool = False,
):
    print(f"Evaluating pruning robustness ({pruning_method}, {pruning_amount})")

    identity = pw.generate_random_identity()
    if pruning_method == "l1-unstructured":
        pruned_model = deepcopy(source)
        insert_res = pw.insert_watermark(pruned_model, identity, verbose=verbose)
        l1_unstructured_prune_model(pruned_model, amount=pruning_amount)
    elif pruning_method == "sparse-gpt":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        pruned_model = deepcopy(source).float()
        insert_res = pw.insert_watermark(pruned_model, identity, verbose=verbose)
        sparse_gpt_unstructured_pruning(
            pruned_model,
            tokenizer,
            dataset.select(range(16)),
            sparsity=1.0 - pruning_amount,
            sample_num=16,
            prune_n=0,
            prune_m=0,
            device="cpu",
        )
    else:
        raise ValueError(f"Invalid pruning method: {pruning_method}")

    extract_res = pw.extract_watermark(source, pruned_model, verbose=verbose)
    diff, total, robustness = compare_watermarks(insert_res, extract_res)
    print(
        f"Pruning ({pruning_method}, {pruning_amount}): {diff}/{total} corrupted digits"
        f"\nRobustness: {robustness}"
    )

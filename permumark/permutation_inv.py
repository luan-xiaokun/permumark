"""Invariant permutation operation on transformer models."""

from __future__ import annotations

import warnings

import torch
from transformers import PreTrainedModel

from .permutation_extraction import (
    PermExtractionResult,
    extract_perm,
    build_cost_matrix,
    solve_perm_by_lsa,
)


class FunctionalInvariantError(Exception):
    """Error raised when generating or extracting functional invariants of a model."""


class InvalidPermutationError(FunctionalInvariantError):
    """Error raised when the permutation is invalid."""


class MismatchPermutationError(FunctionalInvariantError):
    """Error raised when the permutation size does not match the weight size."""


class ExtractionBlockPermutationError(FunctionalInvariantError):
    """Error raised when the extracted permutation is not block permutation."""


def check_tensor_perm(perm: torch.Tensor) -> None:
    """Check if the permutation tensor is valid."""
    if perm.dim() != 1:
        err_msg = f"perm should be 1D tensor, got {perm.dim()}D tensor"
    elif perm.min() < 0:
        err_msg = f"perm should be non-negative, got {perm.min()}"
    elif perm.max() >= perm.size(0):
        err_msg = f"perm should be less than {perm.size(0)}, got {perm.max()}"
    elif len(perm.unique()) != perm.size(0):
        err_msg = "perm should have unique elements"
    else:
        return
    raise InvalidPermutationError(err_msg)


def permute_feed_forward(
    model: PreTrainedModel, perm: list[int] | torch.Tensor, index: int
) -> None:
    """
    Permute the feed forward layer in a transformer model.
    :param model: transformer model to permute
    :param perm: permutation list
    :param index: index of the feed forward layer
    :return: None
    """
    perm = torch.as_tensor(perm)
    check_tensor_perm(perm)

    mlp = model.model.layers[index].mlp
    if perm.size(0) != mlp.gate_proj.weight.data.size(0):
        raise MismatchPermutationError(
            f"Permutation size {perm.size(0)} does not match "
            f"weight size {mlp.gate_proj.weight.data.size(0)}"
        )

    mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[perm, :]
    mlp.up_proj.weight.data = mlp.up_proj.weight.data[perm, :]
    mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, perm]


def permute_embeddings(model: PreTrainedModel, perm: list[int] | torch.Tensor) -> None:
    """
    Permute the embedding weights in a transformer model.
    :param model: transformer model to permute
    :param perm: permutation list
    :return: None
    """
    perm = torch.as_tensor(perm)
    check_tensor_perm(perm)

    et = model.model.embed_tokens
    if perm.size(0) != et.weight.data.size(1):
        raise MismatchPermutationError(
            f"Permutation size {perm.size(0)} does not "
            f"match weight size {et.size(1)}"
        )

    et.weight.data = et.weight.data[:, perm]
    for layer in model.model.layers:
        layer.input_layernorm.weight.data = layer.input_layernorm.weight.data[perm]
        if getattr(layer, "post_attention_layernorm", None) is not None:
            layer.post_attention_layernorm.weight.data = (
                layer.post_attention_layernorm.weight.data[perm]
            )
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, perm]
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, perm]
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[:, perm]
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[perm, :]
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[:, perm]
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[:, perm]
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[perm, :]
    model.model.norm.weight.data = model.model.norm.weight.data[perm]
    if not model.config.tie_word_embeddings:
        model.lm_head.weight.data = model.lm_head.weight.data[:, perm]


def permute_attention_heads(
    model: PreTrainedModel,
    q_perm: list[int] | torch.Tensor,
    kv_perm: list[int] | torch.Tensor,
    index: int,
) -> None:
    """
    Permute the attention heads in a transformer model.
    :param model: transformer model to permute
    :param q_perm: permutation list for query heads
    :param kv_perm: permutation list for key-value heads
    :param index: index of the feed forward layer
    :return: None
    """
    q_perm = torch.as_tensor(q_perm)
    kv_perm = torch.as_tensor(kv_perm)
    check_tensor_perm(q_perm)
    check_tensor_perm(kv_perm)

    attn = model.model.layers[index].self_attn
    attn.q_proj.weight.data = attn.q_proj.weight.data[q_perm, :]
    attn.k_proj.weight.data = attn.k_proj.weight.data[kv_perm, :]
    attn.v_proj.weight.data = attn.v_proj.weight.data[kv_perm, :]
    attn.o_proj.weight.data = attn.o_proj.weight.data[:, q_perm]


def extract_feed_forward_perm(
    source: PreTrainedModel, model: PreTrainedModel, index: int
) -> PermExtractionResult:
    """
    Extract feed forward permutations from a transformer model.
    :param source: source model without permutation
    :param model: model to extract permutations from
    :param index: index of the feed forward layer
    :return: PermExtractionResult
    """
    cost_matrix = build_feed_forward_perm_cost_matrix(source, model, index)
    return solve_feed_forward_perm_by_lsa(source, model, index, cost_matrix)[1]


def build_feed_forward_perm_cost_matrix(
    source: PreTrainedModel, model: PreTrainedModel, index: int
) -> torch.Tensor:
    """
    Build cost matrix for feed forward permutations.
    :param source: source model without permutation
    :param model: model to extract permutations from
    :param index: index of the feed forward layer
    :return: a cost matrix for feed forward permutations
    """
    w1 = source.model.layers[index].mlp.gate_proj.weight.data
    w1_ = model.model.layers[index].mlp.gate_proj.weight.data
    return build_cost_matrix(w1, w1_)


def solve_feed_forward_perm_by_lsa(
    source: PreTrainedModel,
    model: PreTrainedModel,
    index: int,
    cost_matrix: torch.Tensor,
) -> tuple[int, PermExtractionResult]:
    """
    Solve feed forward permutations by linear sum assignment formulation.
    :param source: source model without permutation
    :param model: model to extract permutations from
    :param index: index of the feed forward layer
    :param cost_matrix: cost matrix for feed forward permutations
    :return: PermExtractionResult
    """
    w1 = source.model.layers[index].mlp.gate_proj.weight.data
    w1_ = model.model.layers[index].mlp.gate_proj.weight.data
    cost_matrix = cost_matrix + 1e-8 * torch.rand_like(cost_matrix)
    return index, solve_perm_by_lsa(w1, w1_, cost_matrix)


def extract_embeddings_perm(
    source: PreTrainedModel, model: PreTrainedModel
) -> PermExtractionResult:
    """
    Extract embeddings permutation from a transformer model.
    :param source: source model without permutation
    :param model: model to extract permutations from
    :return: PermExtractionResult
    """
    ew = source.model.embed_tokens.weight.data
    ew_ = model.model.embed_tokens.weight.data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, n = ew.shape
    if n * n * m > 1e12:
        warnings.warn("too large vocab, using CPU")
        device = "cpu"
    return extract_perm(ew, ew_, dim="col", device=device)


def extract_attention_heads_perm(
    source: PreTrainedModel, model: PreTrainedModel, index: int
) -> tuple[PermExtractionResult, PermExtractionResult]:
    """
    Extract attention heads permutation from a transformer model.
    :param source: source model without permutation
    :param model: model to extract permutations from
    :param index: index of the attention layer
    :return: two PermExtractionResult for query and key-value heads
    """
    cost_matrices = build_attention_heads_perm_cost_matrix(source, model, index)
    index, *results = solve_attention_heads_perm_by_lsa(
        source, model, index, cost_matrices
    )
    return results


def build_attention_heads_perm_cost_matrix(
    source: PreTrainedModel, model: PreTrainedModel, index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build attention heads permutation cost matrices.
    :param source: source model without permutation
    :param model: model to extract permutations from
    :param index: index of the attention layer
    :return: two cost matrices for query and key-value heads
    """
    num_attention_heads = getattr(source.config, "num_attention_heads")
    if hasattr(source.config, "head_dim"):
        block_size = int(source.config.head_dim)
    else:
        assert hasattr(source.config, "hidden_size")
        block_size = int(source.config.hidden_size / num_attention_heads)

    wv = source.model.layers[index].self_attn.v_proj.weight.data
    wv_ = model.model.layers[index].self_attn.v_proj.weight.data
    cost_matrix_kv = build_cost_matrix(wv, wv_, block_size=block_size)
    wo = source.model.layers[index].self_attn.o_proj.weight.data
    wo_ = model.model.layers[index].self_attn.o_proj.weight.data
    cost_matrix_q = build_cost_matrix(wo, wo_, dim="col", block_size=block_size)

    return cost_matrix_q, cost_matrix_kv


def solve_attention_heads_perm_by_lsa(
    source: PreTrainedModel,
    model: PreTrainedModel,
    index: int,
    cost_matrices: tuple[torch.Tensor, torch.Tensor],
) -> tuple[int, PermExtractionResult, PermExtractionResult]:
    wv = source.model.layers[index].self_attn.v_proj.weight.data
    wv_ = model.model.layers[index].self_attn.v_proj.weight.data
    wo = source.model.layers[index].self_attn.o_proj.weight.data
    wo_ = model.model.layers[index].self_attn.o_proj.weight.data

    num_attention_heads = getattr(source.config, "num_attention_heads")
    if hasattr(source.config, "head_dim"):
        block_size = int(source.config.head_dim)
    else:
        assert hasattr(source.config, "hidden_size")
        block_size = int(source.config.hidden_size / num_attention_heads)

    cost_matrix_q, cost_matrix_kv = cost_matrices
    cost_matrix_q = cost_matrix_q + 1e-8 * torch.rand_like(cost_matrix_q)
    cost_matrix_kv = cost_matrix_kv + 1e-8 * torch.rand_like(cost_matrix_kv)
    result_q = solve_perm_by_lsa(
        wo, wo_, cost_matrix_q, dim="col", block_size=block_size
    )
    if result_q.block_perm is None:
        raise ExtractionBlockPermutationError(
            f"block_size={result_q.block_size}, perm={result_q.perm}"
        )
    result_kv = solve_perm_by_lsa(wv, wv_, cost_matrix_kv, block_size=block_size)
    if result_kv.block_perm is None:
        raise ExtractionBlockPermutationError(
            f"block_size={result_kv.block_size}, perm={result_kv.perm}"
        )
    return index, result_q, result_kv

"""High-level functions for watermarking and extracting identity from a model."""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import torch
from sympy.functions.combinatorial.factorials import subfactorial
from transformers import PretrainedConfig, PreTrainedModel

from . import ecc, permutation_inv, permutation_mapping, utils


def get_ecc_params(
    num_kv_heads: int,
    group_size: int,
    total_id_num: int,
    max_corrupt_prob: float = 1e-4,
) -> tuple[int, int]:
    """
    Compute parameter settings for RSC, q and k.
    :param num_kv_heads: number of key-value heads in the model
    :param group_size: size of query attention group
    :param total_id_num: total number of different identities
    :param max_corrupt_prob: maximal probability of undetected corruption
    :return: values for q and k
    """
    size = int(subfactorial(num_kv_heads))
    if group_size > 1:
        size *= int(subfactorial(group_size) ** num_kv_heads)
    upper = math.floor(math.log2(max_corrupt_prob * size))
    k = 1
    while True:
        lower = math.ceil(math.log2(total_id_num) / k)
        if lower <= upper:
            gf_size = int(2**lower)
            return gf_size, k
        k += 1


class PermutationWatermarkInsertionResult(NamedTuple):
    """A named tuple to store the result of watermark insertion."""

    identity: list[int]
    watermark: list[int]
    permutations: list[list[int] | tuple[list[int], list[list[int]]]]


class PermutationWatermarkExtractionResult(NamedTuple):
    """A named tuple to store the result of watermark extraction."""

    identity: list[int]
    watermark: list[int]
    erasure_idx: list[int]
    eps_list: list[float]
    time_list: list[float]
    permutations: list[list[int] | tuple[list[int], list[list[int]]]]


class PermutationWatermark:
    """
    Error correction enhanced permutation watermark for transformer models.
    :param config: configuration of the model to watermark
    :param max_corrupt_prob: maximal probability that a corruption not being detected
    :param total_id_num: total number of different identities to manage
    :param evaluation_points: points for evaluation (for code)
    :param column_multipliers: multipliers for columns (for code)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_corrupt_prob: float = 1e-4,
        total_id_num: int = 10_000_000,
        evaluation_points: list[int] | None = None,
        column_multipliers: list[int] | None = None,
    ):
        self.max_corrupt_prob = max_corrupt_prob
        self.total_id_num = total_id_num

        self.layer_num = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", self.num_attention_heads
        )
        self.group_size = self.num_attention_heads // self.num_kv_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_attention_heads
        self.slots = ("embeddings",) + ("attention", "feed_forward") * self.layer_num
        gf_size, id_length = get_ecc_params(
            self.num_kv_heads, self.group_size, total_id_num, max_corrupt_prob
        )
        self.evaluation_points = evaluation_points
        self.column_multipliers = column_multipliers
        self.ecc = ecc.ReedSolomonCode(
            gf_size, len(self.slots), id_length, evaluation_points, column_multipliers
        )
        self.perm_map = permutation_mapping.PermutationMapping(
            gf_size,
            config.hidden_size,
            config.intermediate_size,
            self.num_kv_heads,
            self.group_size,
        )

    def __len__(self) -> int:
        return len(self.slots)

    def __repr__(self) -> str:
        emb_size = str(self.perm_map.sizes["embeddings"])
        ff_size = str(self.perm_map.sizes["feed_forward"])
        attn_size = f"{self.num_kv_heads}x{self.group_size}"
        res = (
            f"{self.__class__.__name__}(q={self.ecc.q}, k={self.ecc.dimension}, "
            f"n={self.ecc.length}, emb={emb_size}, attn={attn_size}, ff={ff_size}, "
            f"max_corrupt_prob={self.max_corrupt_prob}, "
            f"total_id_num={self.total_id_num})"
        )
        return res

    def to_dict(self) -> dict:
        """
        Convert the watermark config to a dictionary.
        :return: configuration dictionary
        """
        return {
            "config": {
                "num_hidden_layers": self.layer_num,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_kv_heads,
                "hidden_size": self.head_dim * self.num_attention_heads,
                "intermediate_size": self.perm_map.sizes["feed_forward"],
            },
            "max_corrupt_prob": self.max_corrupt_prob,
            "total_id_num": self.total_id_num,
            "evaluation_points": self.evaluation_points,
            "column_multipliers": self.column_multipliers,
        }

    @classmethod
    def from_dict(cls, params: dict) -> PermutationWatermark:
        """
        Construct a watermark configuration from a dictionary.
        :param params: parameters of the watermark configuration
        :return: a PermutationWatermark instance
        """
        config = PretrainedConfig(**params["config"])
        return cls(
            config,
            params["max_corrupt_prob"],
            params["total_id_num"],
            params["evaluation_points"],
            params["column_multipliers"],
        )

    def generate_random_identity(self) -> list[int]:
        """
        Generate a random identity message.
        :return: a list of symbols in the finite field
        """
        return [random.randint(0, self.ecc.q - 1) for _ in range(self.ecc.dimension)]

    def insert_watermark(
        self, model: PreTrainedModel, identity: list[int], verbose: bool = False
    ) -> PermutationWatermarkInsertionResult:
        """
        Embed the identity message into the given model.
        :param model: transformer model to insert watermark into
        :param identity: identity message to embed
        :param verbose: verbose output
        :return: a PermutationWatermarkInsertionResult instance
        """
        watermark = self.ecc.encode(identity)
        permutations = list(self.perm_map.encode_codeword(watermark, self.slots))
        if verbose:
            print(f"Inserted identity: {identity}")
            print(f"Inserted watermark: {watermark}")

        with torch.no_grad():
            for i, (perm, slot) in enumerate(zip(permutations, self.slots)):
                if slot == "embeddings":
                    permutation_inv.permute_embeddings(model, perm)
                elif slot == "feed_forward":
                    permutation_inv.permute_feed_forward(model, perm, (i - 1) // 2)
                elif slot == "attention":
                    if isinstance(perm, list):
                        q_perm = kv_perm = utils.block_perm_gen(self.head_dim, perm)
                    elif isinstance(perm, tuple):
                        kv_base_perm, group_perms = perm
                        kv_perm = utils.block_perm_gen(self.head_dim, kv_base_perm)
                        q_perm = utils.block_perm_gen(
                            self.head_dim,
                            torch.cat([torch.as_tensor(gp) for gp in group_perms])
                            + torch.as_tensor(kv_base_perm).repeat_interleave(
                                self.group_size
                            )
                            * self.group_size,
                        )
                    permutation_inv.permute_attention_heads(
                        model, q_perm, kv_perm, (i - 1) // 2
                    )

        return PermutationWatermarkInsertionResult(identity, watermark, permutations)

    def extract_watermark(
        self,
        source: PreTrainedModel,
        model: PreTrainedModel,
        verbose: bool = False,
        max_workers: int = 12,
    ) -> PermutationWatermarkExtractionResult:
        """
        Extract watermark and decode to identity message from models.
        :param source: transformer model without the watermark
        :param model: transformer model with inserted watermark
        :param verbose: verbose output
        :param max_workers: number of workers, -1 means no multithreading
        :return: a PermutationWatermarkExtractionResult instance
        """
        if max_workers <= 0:
            permutations: list = []
            eps_list = []
            time_list = []
            with torch.no_grad():
                for i, slot in enumerate(self.slots):
                    if slot == "embeddings":
                        result = permutation_inv.extract_embeddings_perm(source, model)
                        permutations.append(result.perm)
                        eps_list.append(result.eps_norm)
                        time_list.append(result.time)
                        # NOTE propagated to other layers
                        permutation_inv.permute_embeddings(
                            model, utils.get_perm_inv(result.perm)
                        )
                    elif slot == "feed_forward":
                        result = permutation_inv.extract_feed_forward_perm(
                            source, model, (i - 1) // 2
                        )
                        permutations.append(result.perm)
                        eps_list.append(result.eps_norm)
                        time_list.append(result.time)
                    elif slot == "attention":
                        result_q, result_kv = (
                            permutation_inv.extract_attention_heads_perm(
                                source, model, (i - 1) // 2
                            )
                        )
                        attn_perm = self.decode_attention_qkv_results(
                            result_q, result_kv
                        )
                        permutations.append(attn_perm)
                        eps_list.append(max(result_q.eps_norm, result_kv.eps_norm))
                        time_list.append(max(result_q.time, result_kv.time))
        else:
            permutations: list = [None] * len(self.slots)
            eps_list = [0.0] * len(self.slots)
            time_list = [0.0] * len(self.slots)
            cost_matrices = []
            with torch.no_grad():
                for i, slot in enumerate(self.slots):
                    if slot == "embeddings":
                        result = permutation_inv.extract_embeddings_perm(source, model)
                        permutations[0] = result.perm
                        eps_list[0] = result.eps_norm
                        time_list[0] = result.time
                        # NOTE propagated to other layers
                        permutation_inv.permute_embeddings(
                            model, utils.get_perm_inv(result.perm)
                        )
                    elif slot == "feed_forward":
                        ff_cm = permutation_inv.build_feed_forward_perm_cost_matrix(
                            source, model, (i - 1) // 2
                        )
                        cost_matrices.append(ff_cm)
                    elif slot == "attention":
                        attn_cm = (
                            permutation_inv.build_attention_heads_perm_cost_matrix(
                                source, model, (i - 1) // 2
                            )
                        )
                        cost_matrices.append(attn_cm)
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, cm in enumerate(cost_matrices):
                    if isinstance(cm, torch.Tensor):
                        futures.append(
                            executor.submit(
                                permutation_inv.solve_feed_forward_perm_by_lsa,
                                *(source, model, i // 2, cm),
                            )
                        )
                    else:
                        assert isinstance(cm, tuple) and len(cm) == 2
                        futures.append(
                            executor.submit(
                                permutation_inv.solve_attention_heads_perm_by_lsa,
                                *(source, model, i // 2, cm),
                            )
                        )
                for future in as_completed(futures):
                    index, *result = future.result()
                    if len(result) == 1:
                        permutations[2 * index + 2] = result[0].perm
                        eps_list[2 * index + 2] = result[0].eps_norm
                        time_list[2 * index + 2] = result[0].time
                    else:
                        result_q, result_kv = result
                        attn_perm = self.decode_attention_qkv_results(
                            result_q, result_kv
                        )
                        permutations[2 * index + 1] = attn_perm
                        eps_list[2 * index + 1] = max(
                            result_q.eps_norm, result_kv.eps_norm
                        )
                        time_list[2 * index + 1] = max(result_q.time, result_kv.time)

        watermark = list(self.perm_map.decode_perms(permutations, self.slots))
        identity, erasure_idx = self.decode_extracted_watermark(watermark)
        if verbose:
            print(f"Extracted identity: {identity}")
            print(f"Extracted watermark: {watermark}")
            print(f"Erasure index: {erasure_idx}")
            print(f"Average eps: {sum(eps_list) / len(eps_list):.6f}")

        return PermutationWatermarkExtractionResult(
            identity, watermark, erasure_idx, eps_list, time_list, permutations
        )

    def decode_extracted_watermark(
        self, watermark: list[int | None]
    ) -> tuple[list[int] | None, list[int]]:
        """
        Decode extracted watermark to get identity message.
        :param watermark: extracted watermark, may contain erasures
        :return: a tuple of decoded identity and erasure indices
        """
        watermark_ = [0] * len(watermark)
        erasure_idx = []
        for i, digit in enumerate(watermark):
            if digit is not None and 0 <= digit < self.ecc.q:
                watermark_[i] = digit
            else:
                erasure_idx.append(i)
        return self.ecc.decode(watermark_, erasure_idx), erasure_idx

    def decode_attention_qkv_results(
        self,
        result_q: permutation_inv.PermExtractionResult,
        result_kv: permutation_inv.PermExtractionResult,
    ) -> list[int] | tuple:
        kv_base_perm = utils.extract_block_permutation(
            result_kv.block_perm, self.head_dim
        )
        assert isinstance(kv_base_perm, list)
        eps = max(result_q.eps_norm, result_kv.eps_norm)
        time = max(result_q.time, result_kv.time)
        if self.group_size == 1:
            return kv_base_perm
        group_perms = []
        stride = self.head_dim * self.group_size
        for j in range(0, len(result_q.block_perm), stride):
            block = torch.as_tensor(result_q.block_perm[j : j + stride])
            block = block - block.min()
            group_perm = utils.extract_block_permutation(block, self.head_dim)
            if isinstance(group_perm, tuple):
                raise permutation_inv.ExtractionBlockPermutationError(
                    f"block={group_perm[0]}, perm={group_perm[1]}"
                )
            group_perms.append(group_perm)
        return kv_base_perm, group_perms

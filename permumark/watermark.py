"""High-level functions for watermarking and extracting identity from a model."""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple

import torch
import tqdm
from sympy.functions.combinatorial.factorials import subfactorial
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half

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


class TransformerWatermark(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, params: dict) -> TransformerWatermark:
        pass

    @abstractmethod
    def generate_random_identity(self) -> list[int]:
        pass

    @abstractmethod
    def insert_watermark(
        self,
        model: PreTrainedModel,
        identity: list[int],
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def extract_watermark(
        self, source: PreTrainedModel, model: PreTrainedModel, **kwargs
    ) -> Any:
        pass


class PermutationWatermark(TransformerWatermark):
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
        max_workers: int = 8,
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
        """
        Decode extracted permutations in attention heads.
        :param result_q: extracted permutation on query weights
        :param result_kv: extracted permutation on key/value weights
        :return: the base permutation on key/value heads, and permutations for
        all query attention groups
        """
        kv_base_perm = utils.extract_block_permutation(
            result_kv.block_perm, self.head_dim
        )
        assert isinstance(kv_base_perm, list)
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


class FunctionInvariantTransformationWatermark(TransformerWatermark):
    """
    Function invariant transformation watermark for transformer models.
    :param config: configuration of the model to watermark
    :param mode: settings of the FIT watermark, four modes, including permutation,
    scaling, qk-products, and all
    """

    def __init__(self, config: PretrainedConfig, mode: str):
        assert mode in ("permutation", "scaling", "qk-products", "all")
        self.mode = mode
        self.layer_num = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", self.num_attention_heads
        )
        self.group_size = self.num_attention_heads // self.num_kv_heads
        if self.group_size > 1:
            raise ValueError("Do not support grouped-query attention!")
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_attention_heads
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.slot_choice_num = 2**8
        self.candidates: dict[str, list[list[torch.Tensor]]] = {
            "permutation": [],
            "scaling": [],
            "qk-products": [],
        }

        if self.mode == "permutation" or self.mode == "all":
            self.candidates["permutation"].append(
                self._construct_slot_candidates(perm_size=config.hidden_size)
            )
            for _ in range(self.layer_num):
                self.candidates["permutation"].append(
                    self._construct_slot_candidates(perm_size=self.num_kv_heads)
                )
                self.candidates["permutation"].append(
                    self._construct_slot_candidates(perm_size=config.intermediate_size)
                )
        if self.mode == "scaling" or self.mode == "all":
            for _ in range(2 * self.layer_num):
                self.candidates["scaling"].append(
                    self._construct_slot_candidates(scaling_size=config.hidden_size)
                )
        if self.mode == "qk-products" or self.mode == "all":
            for _ in range(self.layer_num):
                self.candidates["qk-products"].append(
                    self._construct_slot_candidates(
                        qk_prod_size=(self.head_dim, self.num_kv_heads)
                    )
                )
        print(
            f"Generated {len(self) * self.slot_choice_num} candidate transformations "
            f"for watermarking ({self.mode} mode)"
        )

    def _construct_slot_candidates(
        self,
        perm_size: int | None = None,
        scaling_size: int | None = None,
        qk_prod_size: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        def make_2d_rotary_degrees(head_dim, num_kv_heads):
            theta = 2 * torch.pi * torch.rand(head_dim // 2)
            return theta.repeat(2).repeat(num_kv_heads, 1)

        if perm_size is not None:
            return [torch.randperm(perm_size) for _ in range(self.slot_choice_num)]
        if scaling_size is not None:
            return [
                10 ** (torch.rand(scaling_size) * 2 - 1)
                for _ in range(self.slot_choice_num)
            ]
        if qk_prod_size is not None:
            return [
                make_2d_rotary_degrees(*qk_prod_size)
                for _ in range(self.slot_choice_num)
            ]

        raise ValueError("Unknown mode")

    def __len__(self) -> int:
        return sum(map(len, self.candidates.values()))

    def __repr__(self) -> str:
        attn_size = f"{self.num_kv_heads}x{self.group_size}"
        res = (
            f"{self.__class__.__name__}(n={len(self)}, emb={self.hidden_size}, "
            f"attn={attn_size}, ff={self.intermediate_size})"
        )
        return res

    def to_dict(self) -> dict:
        """
        Convert the watermark config to a dictionary.
        :return: configuration dictionary
        """
        params = {
            "config": {
                "num_hidden_layers": self.layer_num,
                "num_attention_heads": self.num_attention_heads,
                "num_kv_heads": self.num_kv_heads,
                "hidden_size": self.head_dim * self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
            },
            "slot_choice_num": self.slot_choice_num,
            "mode": self.mode,
            "candidates": {
                "permutation": [],
                "scaling": [],
                "qk-products": [],
            },
        }
        for cand_type, cand_values in self.candidates.items():
            params["candidates"][cand_type] = [
                [cand.tolist() for cand in values] for values in cand_values
            ]
        return params

    @classmethod
    def from_dict(cls, params: dict) -> FunctionInvariantTransformationWatermark:
        """
        Construct a FIT instance from a configuration dictionary.
        :param params: parameters of the watermark configuration
        :return: a FunctionInvariantTransformationWatermark instance
        """
        config = PretrainedConfig(**params["config"])
        res = cls(
            config,
            params["mode"],
        )
        for cand_type, cand_values in params["candidates"].items():
            res.candidates[cand_type] = [
                [torch.tensor(cand) for cand in values] for values in cand_values
            ]
        return res

    def generate_random_identity(self) -> list[int]:
        """
        Generate a random model identifier.
        :return: a list of integers representing the watermark
        """
        identity = []
        if self.mode in ("permutation", "all"):
            identity.extend(
                [
                    random.randint(0, self.slot_choice_num - 1)
                    for _ in range(2 * self.layer_num + 1)
                ]
            )
        if self.mode in ("scaling", "all"):
            identity.extend(
                [
                    random.randint(0, self.slot_choice_num - 1)
                    for _ in range(2 * self.layer_num)
                ]
            )
        if self.mode in ("qk-products", "all"):
            identity.extend(
                [
                    random.randint(0, self.slot_choice_num - 1)
                    for _ in range(self.layer_num)
                ]
            )
        return identity

    def insert_watermark(
        self,
        model: PreTrainedModel,
        identity: list[int],
        **kwargs,
    ) -> Any:
        """
        Embed the model identifier into the given model.
        :param model: transformer model to insert watermark into
        :param identity: identity message to embed
        :param kwargs: other arguments
        :return: None
        """
        id_copy = identity[::-1]

        print("Applying dimension permutations")
        if self.mode in ("permutation", "all"):
            with torch.no_grad():
                digit = id_copy.pop()
                perm = self.candidates["permutation"][0][digit]
                permutation_inv.permute_embeddings(model, perm)
                for i in range(self.layer_num):
                    digit = id_copy.pop()
                    perm = self.candidates["permutation"][2 * i + 1][digit]
                    block_perm = utils.block_perm_gen(self.head_dim, perm)
                    permutation_inv.permute_attention_heads(
                        model, block_perm, block_perm, i
                    )
                    digit = id_copy.pop()
                    perm = self.candidates["permutation"][2 * i + 2][digit]
                    permutation_inv.permute_feed_forward(model, perm, i)

        print("Applying scaling/unscaling")
        if self.mode in ("scaling", "all"):
            for i in range(self.layer_num):
                layer = model.model.layers[i]
                dtype = layer.post_attention_layernorm.weight.data.dtype
                with torch.no_grad():
                    digit = id_copy.pop()
                    alpha = self.candidates["scaling"][2 * i][digit].to(dtype)
                    layer.input_layernorm.weight.data = (
                        alpha * layer.input_layernorm.weight.data
                    )
                    layer.self_attn.q_proj.weight.data = (
                        1.0 / alpha * layer.self_attn.q_proj.weight.data
                    )
                    layer.self_attn.k_proj.weight.data = (
                        1.0 / alpha * layer.self_attn.k_proj.weight.data
                    )
                    layer.self_attn.v_proj.weight.data = (
                        1.0 / alpha * layer.self_attn.v_proj.weight.data
                    )
                    digit = id_copy.pop()
                    alpha = self.candidates["scaling"][2 * i + 1][digit].to(dtype)
                    layer.post_attention_layernorm.weight.data = (
                        alpha * layer.post_attention_layernorm.weight.data
                    )
                    layer.mlp.up_proj.weight.data = (
                        1.0 / alpha * layer.mlp.up_proj.weight.data
                    )
                    layer.mlp.gate_proj.weight.data = (
                        1.0 / alpha * layer.mlp.gate_proj.weight.data
                    )

        print("Applying invertible matrices in QK products")
        if self.mode in ("qk-products", "all"):
            with torch.no_grad():
                for i in range(self.layer_num):
                    attn = model.model.layers[i].self_attn
                    digit = identity[-(self.layer_num - i)]
                    dtype = attn.q_proj.weight.data.dtype
                    theta = self.candidates["qk-products"][i][digit]
                    wq, wk = apply_qk_products(
                        theta,
                        attn.q_proj.weight.data,
                        attn.q_proj.weight.data,
                        self.head_dim,
                        self.num_kv_heads,
                        dtype,
                    )
                    attn.q_proj.weight.data = wq.cpu()
                    attn.k_proj.weight.data = wk.cpu()

    def extract_watermark(
        self, source: PreTrainedModel, model: PreTrainedModel, **kwargs
    ) -> tuple[list[int], tuple[float, float, float]]:
        """
        Extract the watermark from the given model and the source model.
        :param source: transformer model without the watermark
        :param model: transformer model with inserted watermark
        :param kwargs: other arguments
        :return: the extracted watermark, a list of integers
        """
        extracted_identity = []

        perm_extract_start = time.time()
        if self.mode in ("permutation", "all"):
            with torch.no_grad():
                weight = source.model.embed_tokens.weight.data
                weight_ = model.model.embed_tokens.weight.data
                dists = [
                    torch.norm(weight_ - weight[:, perm])
                    for perm in self.candidates["permutation"][0]
                ]
                digit = torch.tensor(dists).argmin().item()
                extracted_identity.append(digit)
                perm_inv = utils.get_perm_inv(
                    self.candidates["permutation"][0][digit].tolist()
                )
                permutation_inv.permute_embeddings(model, perm_inv)

                for i in tqdm.tqdm(
                    range(self.layer_num), desc="Extracting permutation"
                ):
                    weight = source.model.layers[i].self_attn.o_proj.weight.data
                    weight_ = model.model.layers[i].self_attn.o_proj.weight.data
                    dists = [
                        torch.norm(
                            weight_
                            - weight[:, utils.block_perm_gen(self.head_dim, perm)]
                        )
                        for perm in self.candidates["permutation"][2 * i + 1]
                    ]
                    digit = torch.tensor(dists).argmin().item()
                    extracted_identity.append(digit)
                    # apply the inverse permutation
                    inv_perm = utils.get_perm_inv(
                        self.candidates["permutation"][2 * i + 1][digit].tolist()
                    )
                    block_inv_perm = utils.block_perm_gen(self.head_dim, inv_perm)
                    permutation_inv.permute_attention_heads(
                        model, block_inv_perm, block_inv_perm, i
                    )

                    weight = source.model.layers[i].mlp.gate_proj.weight.data
                    weight_ = model.model.layers[i].mlp.gate_proj.weight.data
                    dists = [
                        torch.norm(weight_ - weight[perm, :])
                        for perm in self.candidates["permutation"][2 * i + 2]
                    ]
                    digit = torch.tensor(dists).argmin().item()
                    extracted_identity.append(digit)
                    # apply the inverse permutation
                    inv_perm = utils.get_perm_inv(
                        self.candidates["permutation"][2 * i + 2][digit].tolist()
                    )
                    permutation_inv.permute_feed_forward(model, inv_perm, i)

        scaling_extract_start = time.time()
        if self.mode in ("scaling", "all"):
            with torch.no_grad():
                for i in range(self.layer_num):
                    layer = model.model.layers[i]
                    dtype = model.model.layers[i].self_attn.q_proj.weight.data.dtype
                    weight = source.model.layers[i].self_attn.v_proj.weight.data.cuda()
                    weight_ = model.model.layers[i].self_attn.v_proj.weight.data.cuda()
                    dists = [
                        torch.norm(weight_ - 1.0 / alpha.cuda() * weight)
                        for alpha in self.candidates["scaling"][2 * i]
                    ]
                    digit = torch.tensor(dists).argmin().item()
                    extracted_identity.append(digit)
                    # multiply the scaling factor
                    alpha = self.candidates["scaling"][2 * i][digit].to(dtype)
                    layer.input_layernorm.weight.data = (
                        1.0 / alpha * layer.input_layernorm.weight.data
                    )
                    layer.self_attn.q_proj.weight.data = (
                        alpha * layer.self_attn.q_proj.weight.data
                    )
                    layer.self_attn.k_proj.weight.data = (
                        alpha * layer.self_attn.k_proj.weight.data
                    )
                    layer.self_attn.v_proj.weight.data = (
                        alpha * layer.self_attn.v_proj.weight.data
                    )

                    weight = source.model.layers[i].mlp.up_proj.weight.data.cuda()
                    weight_ = model.model.layers[i].mlp.up_proj.weight.data.cuda()
                    dists = [
                        torch.norm(weight_ - 1.0 / alpha.cuda() * weight)
                        for alpha in self.candidates["scaling"][2 * i + 1]
                    ]
                    digit = torch.tensor(dists).argmin().item()
                    extracted_identity.append(digit)
                    # multiply the scaling factor
                    alpha = self.candidates["scaling"][2 * i + 1][digit].to(dtype)
                    layer.post_attention_layernorm.weight.data = (
                        1.0 / alpha * layer.post_attention_layernorm.weight.data
                    )
                    layer.mlp.up_proj.weight.data = (
                        alpha * layer.mlp.up_proj.weight.data
                    )
                    layer.mlp.gate_proj.weight.data = (
                        alpha * layer.mlp.gate_proj.weight.data
                    )
                    torch.cuda.empty_cache()

        qk_products_extract_start = time.time()
        if self.mode in ("qk-products", "all"):
            with torch.no_grad():
                for i in range(self.layer_num):
                    weight = source.model.layers[i].self_attn.q_proj.weight.data.cuda()
                    weight_ = model.model.layers[i].self_attn.q_proj.weight.data.cuda()
                    dtype = weight.dtype
                    dists = [
                        torch.norm(
                            weight_
                            - apply_half_qk_products(
                                theta, weight, self.head_dim, self.num_kv_heads, dtype
                            )
                        )
                        for theta in self.candidates["qk-products"][i]
                    ]
                    extracted_identity.append(torch.tensor(dists).argmin().item())
                    torch.cuda.empty_cache()

        extract_end = time.time()
        perm_time = scaling_extract_start - perm_extract_start
        scaling_time = qk_products_extract_start - scaling_extract_start
        qk_products_time = extract_end - qk_products_extract_start

        return extracted_identity, (perm_time, scaling_time, qk_products_time)


def apply_qk_products(
    theta: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    head_dim: int,
    num_kv_heads: int,
    dtype: torch.dtype,
):
    theta = theta.cuda().to(dtype)
    wq, wk = apply_rotary_pos_emb(
        wq.cuda().view(num_kv_heads, head_dim, -1).transpose(1, 2),
        wk.cuda().view(num_kv_heads, head_dim, -1).transpose(1, 2),
        theta.cos(),
        theta.sin(),
    )
    wq = wq.transpose(1, 2).reshape(head_dim * num_kv_heads, -1)
    wk = wk.transpose(1, 2).reshape(head_dim * num_kv_heads, -1)
    return wq, wk


def apply_half_qk_products(
    theta: torch.Tensor,
    w: torch.Tensor,
    head_dim: int,
    num_kv_heads: int,
    dtype: torch.dtype,
):
    theta = theta.cuda().to(dtype)
    cos = theta.cos().unsqueeze(1)
    sin = theta.sin().unsqueeze(1)
    w = w.cuda().view(num_kv_heads, head_dim, -1).transpose(1, 2)
    w = (w * cos) + (rotate_half(w) * sin)
    w = w.transpose(1, 2).reshape(head_dim * num_kv_heads, -1)
    return w

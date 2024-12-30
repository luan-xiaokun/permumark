"""Attacks for PermuMark."""

from __future__ import annotations

import random

import torch
from sympy.functions.combinatorial.factorials import subfactorial, factorial
from transformers import PreTrainedModel

from permumark import PermutationWatermark, permutation_inv, utils
from permumark.derangement import derangement_unrank


def make_permutation(size: int, perm_type: str = "random") -> torch.Tensor:
    """
    Generate a permutation of given size according to specific type.
    :param size: size of the permutation
    :param perm_type: six types of permutations are supported, including
    - random: a randomly sampled permutation
    - swap: a permutation with a random pair swap
    - shift: a permutation that shifts the sequence by some random offset
    - derangement: a randomly sampled derangement
    - hybrid: randomly use one of the above types
    :return: a permutation in tensor form
    """
    if perm_type == "random":
        perm = torch.randperm(size)
        while torch.equal(perm, torch.arange(size)):
            perm = torch.randperm(size)
        return perm
    if perm_type == "swap":
        perm = torch.arange(size)
        # get two random indices and swap them
        i = j = random.randint(0, size - 1)
        while j == i:
            j = random.randint(0, size - 1)
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    if perm_type == "shift":
        perm = torch.arange(size)
        i = random.randint(1, size - 1)
        return torch.cat((perm[i:], perm[:i]))
    if perm_type == "derangement":
        i = random.randint(0, int(subfactorial(size)) - 1)
        perm = derangement_unrank(i, size)
        return torch.as_tensor(perm)
    if perm_type == "hybrid":
        return make_permutation(
            size, random.choice(["random", "swap", "shift", "derangement"])
        )

    raise ValueError(f"Invalid permutation perm_type '{perm_type}'")


def make_2d_rotation_diagonal_mat(
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a diagonal matrix with 2x2 scaled rotation blocks.
    :param hidden_size: size of the diagonal block
    :return a diagonal matrix
    """
    lam = (2.0 * torch.rand(hidden_size // 2) - 1.0).exp()
    theta = torch.pi * torch.rand(hidden_size // 2)

    cos = torch.cos(theta)
    sin = torch.sin(theta)
    rotation_blocks = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)],
        dim=-2,
    )

    mat = torch.block_diag(*(lam[:, None, None] * rotation_blocks))
    mat_t_inv = torch.block_diag(*(1.0 / lam[:, None, None] * rotation_blocks))

    return mat, mat_t_inv


def simulate_attacks(
    pw: PermutationWatermark, perm_budget: int, perm_type: str
) -> tuple[bool, int, int]:
    """
    Simulate attacks against PermuMark. By simulation, the permutations are not applied
    to the model, but are assumed to be extracted successfully.
    Currently, the strategy of the attack is to corrupt each permutation within budget.
    :param pw: the PermuMark to attack
    :param perm_budget: budget for permutation corruption attacks
    :param perm_type: type of permutations for attacks
    :return: a Boolean indicating whether the attack succeeded, the number of
    detected erasures during extraction, and the number of all corrupted permutations.
    """
    pm = pw.perm_map
    identity = pw.generate_random_identity()
    watermark = pw.ecc.encode(identity)
    perms = list(pm.encode_codeword(watermark, pw.slots))

    # attack perms
    atk_indices = random.sample(range(len(perms)), perm_budget)
    for i, perm in enumerate(perms):
        if i not in atk_indices:
            continue
        if isinstance(perm, list):
            atk_perm = make_permutation(len(perm), perm_type)
            perms[i] = [perm[j] for j in atk_perm]
        elif isinstance(perm, tuple):
            kv_base_perm, group_perms = perm
            atk_kv_perm = make_permutation(len(kv_base_perm), perm_type)
            atk_group_perms = [
                make_permutation(len(gp), perm_type) for gp in group_perms
            ]
            perms[i] = (
                [kv_base_perm[j] for j in atk_kv_perm],
                [
                    [gp[j] for j in atk_p]
                    for gp, atk_p in zip(group_perms, atk_group_perms)
                ],
            )

    extracted_watermark = list(pm.decode_perms(perms, pw.slots))
    extracted_identity, erasure_dix = pw.decode_extracted_watermark(extracted_watermark)
    total_diff = sum(w != ew for w, ew in zip(watermark, extracted_watermark))

    return identity != extracted_identity, len(erasure_dix), total_diff


def scaling_attack(model: PreTrainedModel, index: int) -> None:
    """
    Scaling/unscaling attack on Llama model.
    Two layer norms are scaled by a random factor.
    :param model: the Llama model to attack
    :param index: index of the layer to attack
    :return None
    """
    layer = model.model.layers[index]
    hidden_size = model.config.hidden_size
    dtype = layer.post_attention_layernorm.weight.data.dtype

    # input layernorm
    mu = 10 ** (2.0 * torch.rand(hidden_size).to(dtype) - 1.0)

    layer.input_layernorm.weight.data = mu * layer.input_layernorm.weight.data

    layer.self_attn.q_proj.weight.data = 1.0 / mu * layer.self_attn.q_proj.weight.data
    layer.self_attn.k_proj.weight.data = 1.0 / mu * layer.self_attn.k_proj.weight.data
    layer.self_attn.v_proj.weight.data = 1.0 / mu * layer.self_attn.v_proj.weight.data

    # post attention layernorm
    mu = 10 ** (2.0 * torch.rand(hidden_size).to(dtype) - 1.0)

    layer.post_attention_layernorm.weight.data = (
        mu * layer.post_attention_layernorm.weight.data
    )

    layer.mlp.up_proj.weight.data = 1.0 / mu * layer.mlp.up_proj.weight.data
    layer.mlp.gate_proj.weight.data = 1.0 / mu * layer.mlp.gate_proj.weight.data


def invertible_qk_attack(model: PreTrainedModel, index: int) -> None:
    """
    Invertible matrices in QK products attack on Llama model.
    Applicable to both Llama2 and Llama3 models.
    :param model: the Llama model to attack
    :param index: index of the layer to attack
    :return None
    """
    attn = model.model.layers[index].self_attn

    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_kv_repeat = num_attention_heads // num_key_value_heads
    hidden_size = model.config.hidden_size

    if num_kv_repeat != 1:
        # this is a Llama3 model, using grouped query attention heads
        head_dim = hidden_size // num_attention_heads
        mat_k, mat_k_t_inv = make_2d_rotation_diagonal_mat(hidden_size // num_kv_repeat)
        sub_matrices = [
            mat_k_t_inv[
                i * head_dim : (i + 1) * head_dim, i * head_dim : (i + 1) * head_dim
            ]
            for i in range(num_key_value_heads)
        ]

        repeated_blocks = [
            torch.block_diag(*([sub_mat] * num_kv_repeat)) for sub_mat in sub_matrices
        ]
        mat_q = torch.block_diag(*repeated_blocks)
    else:
        # otherwise, we have num_kv_repeat = 1, this is a Llama2 model
        mat_k, mat_q = make_2d_rotation_diagonal_mat(hidden_size)

    mat_q = mat_q.to(attn.q_proj.weight.data.dtype).cuda()
    mat_k = mat_k.to(attn.k_proj.weight.data.dtype).cuda()

    # apply the attack
    attn.q_proj.weight.data = (mat_q @ attn.q_proj.weight.data.cuda()).cpu()
    attn.k_proj.weight.data = (mat_k @ attn.k_proj.weight.data.cuda()).cpu()


def attack_watermarked_model(
    model: PreTrainedModel,
    pw: PermutationWatermark,
    perm_budget: int,
    scale_attack: bool,
    inv_attack: bool,
    perm_type: str,
) -> None:
    """
    Attack a watermarked model by applying corruptions, scaling attacks, and query-key
    invertible matrices attacks.
    :param model: the watermarked model
    :param pw: a PermutationWatermark instance
    :param perm_budget: number of corruptions on permutations to apply
    :param scale_attack: whether apply scaling attacks
    :param inv_attack: whether apply query-key invertible matrices attacks
    :param perm_type: type of permutations to apply
    :return: None
    """
    layer_num = len(model.model.layers)

    with torch.no_grad():
        if scale_attack:
            for idx in range(layer_num):
                scaling_attack(model, idx)

        if inv_attack:
            for idx in range(layer_num):
                invertible_qk_attack(model, idx)

        atk_indices = random.sample(range(len(pw)), perm_budget)

        for i, slot in enumerate(pw.slots):
            if i not in atk_indices:
                continue
            idx = (i - 1) // 2
            if slot == "embeddings":
                perm = make_permutation(
                    model.model.embed_tokens.weight.data.size(1), perm_type
                )
                permutation_inv.permute_embeddings(model, perm)
            if slot == "feed_forward":
                perm = make_permutation(
                    model.model.layers[idx].mlp.gate_proj.weight.size(0), perm_type
                )
                permutation_inv.permute_feed_forward(model, perm, idx)
            elif slot == "attention":
                if pw.group_size == 1:
                    base_perm = make_permutation(pw.num_kv_heads, perm_type)
                    q_perm = kv_perm = utils.block_perm_gen(pw.head_dim, base_perm)
                else:
                    kv_base_perm = make_permutation(pw.num_kv_heads, perm_type)
                    group_perms = [
                        make_permutation(pw.group_size, perm_type)
                        for _ in range(pw.num_kv_heads)
                    ]
                    kv_perm = utils.block_perm_gen(pw.head_dim, kv_base_perm)
                    q_perm = utils.block_perm_gen(
                        pw.head_dim,
                        torch.cat([torch.as_tensor(gp) for gp in group_perms])
                        + torch.as_tensor(kv_base_perm).repeat_interleave(pw.group_size)
                        * pw.group_size,
                    )
                permutation_inv.permute_attention_heads(model, q_perm, kv_perm, idx)

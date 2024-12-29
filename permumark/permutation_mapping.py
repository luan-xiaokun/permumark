"""Mapping between permutations and integers."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator

from sympy.functions.combinatorial.factorials import subfactorial

from .derangement import derangement_rank, derangement_unrank
from .utils import find_closest_prime, solve_modulo_equations


class PermutationMapping:
    """
    A bijective mapping between permutations and integers (in GF).
    :param gf_size: size of the finite field GF
    :param hidden_dim: hidden dimension at embedding layer
    :param inter_dim: intermediate dimension
    :param num_kv_heads: number of key-value heads
    :param group_size: size of query attention group
    """

    def __init__(
        self,
        gf_size: int,
        hidden_dim: int,
        inter_dim: int,
        num_kv_heads: int,
        group_size: int,
    ) -> None:
        self.gf_size = gf_size
        self.group_size = group_size

        if group_size == 1:
            attn_alpha = self._get_alpha_coefficient(num_kv_heads, gf_size)
            prime1, prime2 = -1, -1
        elif group_size > 1:
            attn_alpha, prime1, prime2 = self._get_beta_coefficient_aux(
                num_kv_heads, group_size, gf_size
            )
        else:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.alphas = {
            "embeddings": self._get_alpha_coefficient(hidden_dim, gf_size),
            "feed_forward": self._get_alpha_coefficient(inter_dim, gf_size),
            "attention": attn_alpha,
            "prime1": prime1,
            "prime2": prime2,
        }

        self.sizes = {
            "embeddings": hidden_dim,
            "feed_forward": inter_dim,
            "attention": num_kv_heads,
        }

    @staticmethod
    def _get_alpha_coefficient(dimension: int, gf_size: int) -> int:
        return int(math.floor(subfactorial(dimension) / gf_size))

    @staticmethod
    def _get_beta_coefficient_aux(
        num_kv_heads: int, group_size: int, gf_size: int
    ) -> tuple[int, int, int]:
        bang_group_size = subfactorial(group_size)
        bang_group_size_exp = bang_group_size**num_kv_heads
        bang_hkv = subfactorial(num_kv_heads)

        prime1 = find_closest_prime(int(bang_group_size_exp))
        prime2 = find_closest_prime(int(bang_hkv))
        if math.gcd(prime1, prime2) != 1:
            prime2 = find_closest_prime(prime2 - 1)
            assert math.gcd(prime1, prime2) == 1
        beta = max(1, prime1 * prime2 // gf_size)
        return beta, prime1, prime2

    def encode(
        self, digit: int, slot_type: str
    ) -> list[int] | tuple[list[int], list[list[int]]]:
        """
        Encode digits in the codeword into permutations.
        Note that the returned permutations are not expanded to block permutations.
        :param digit: a symbol (integer) from the finite field GF
        :param slot_type: type of permutation to insert
        :return: a permutation for embeddings & feed forward slot, or
        a key-value head permutation with a list of group permutations.
        """
        if self.group_size == 1 or slot_type != "attention":
            dimension = self.sizes[slot_type]
            alpha = self.alphas[slot_type]
            return derangement_unrank(alpha * digit, dimension)

        assert slot_type == "attention" and self.group_size > 1
        beta = self.alphas["attention"]
        scaled_digit = beta * digit
        rem1 = scaled_digit % self.alphas["prime1"]
        rem2 = scaled_digit % self.alphas["prime2"]

        kv_base_perm = derangement_unrank(rem2, self.sizes["attention"])

        bang_group_size = int(subfactorial(self.group_size))
        group_digits = []
        while rem1 > 0:
            rem1, gd = divmod(rem1, bang_group_size)
            group_digits.append(gd)
        leading_zero_num = self.sizes["attention"] - len(group_digits)
        group_digits = [0] * leading_zero_num + group_digits[::-1]
        assert all(0 <= d < bang_group_size for d in group_digits)
        group_perms = [derangement_unrank(d, self.group_size) for d in group_digits]
        return kv_base_perm, group_perms

    def decode(
        self, perm: list[int] | tuple[list[int], list[list[int]]], slot_type: str
    ) -> int | None:
        """
        Decode permutations into symbols in the finite field GF.
        Note that input permutations must not be block permutations.
        :param perm: a permutation for embeddings & feed forward slot, or
        a key-value head permutation with a list of group permutations.
        :param slot_type: type of permutation extracted.
        :return: an integer or None if the input is invalid, i.e., an erasure.
        """
        if self.group_size == 1 or slot_type != "attention":
            alpha = self.alphas[slot_type]
            rank = derangement_rank(perm)
            if rank is not None and rank % alpha == 0:
                return int(rank // alpha)
            return None

        assert slot_type == "attention" and self.group_size > 1
        assert len(perm) > 0 and isinstance(perm, tuple) and len(perm) == 2
        beta = self.alphas[slot_type]
        kv_base_perm, group_perms = perm

        # recover rem2
        rem2 = derangement_rank(kv_base_perm)
        if rem2 is None or rem2 >= self.alphas["prime2"]:
            return None

        # recover rem1
        bang_group_size = int(subfactorial(self.group_size))
        group_digits = []
        for group_perm in group_perms:
            gd = derangement_rank(group_perm)
            if gd is None:
                return None
            group_digits.append(gd)
        rem1 = sum(d * bang_group_size**i for i, d in enumerate(group_digits[::-1]))
        if rem1 >= self.alphas["prime1"]:
            return None

        # apply Chinese remainder theorem
        digit = solve_modulo_equations(
            rem1, rem2, self.alphas["prime1"], self.alphas["prime2"]
        )
        if digit % beta != 0:
            return None
        digit = int(digit // beta)

        return digit

    def encode_codeword(
        self, codeword: list[int], slot_types: list[str]
    ) -> Iterator[list[int] | tuple[list[int], list[list[int]]]]:
        with ProcessPoolExecutor() as executor:
            return executor.map(self.encode, codeword, slot_types)

    def decode_perms(
        self,
        perms: list[list[int] | tuple[list[int], list[list[int]]]],
        slot_types: list[str],
    ) -> Iterator[int | None]:
        with ProcessPoolExecutor() as executor:
            return executor.map(self.decode, perms, slot_types)

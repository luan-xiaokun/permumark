"""Utility functions for permutation related operations."""

from __future__ import annotations

import torch


def extract_block_permutation(
    perm: list[int] | torch.Tensor, block_size: int
) -> list[int] | tuple[int, torch.Tensor]:
    """Extract block permutation from a permutation, if it is.
    :param perm: the permutation list
    :param block_size: size of the block
    :return the block permutation list or a tuple of the block index and the
    sub-permutation that makes it not block permutation
    """
    perm_ = []
    for i in range(0, len(perm), block_size):
        sub_perm = torch.as_tensor(perm[i : i + block_size])
        if not torch.equal(sub_perm - sub_perm.min(), torch.arange(block_size)):
            return i, sub_perm
        perm_.append(int(sub_perm.min().item()) // block_size)
    return perm_


def get_perm_inv(perm: list[int]) -> list[int]:
    """Get the inverse permutation of a permutation.
    :param perm: the permutation list
    :return the inverse permutation
    """
    inv_perm = [0] * len(perm)
    for i, ai in enumerate(perm):
        inv_perm[ai] = i
    return inv_perm


def block_perm_gen(block_size: int, perm: list[int] | torch.Tensor) -> torch.Tensor:
    """Generate block permutation.
    :param block_size: block size
    :param perm: base permutation
    :return a block permutation
    """
    base = torch.arange(block_size).repeat(len(perm))
    res = base + torch.as_tensor(perm).repeat_interleave(block_size) * block_size
    return res


def find_closest_prime(n: int) -> int:
    """Given an integer n, find the closest prime number that is less than n.
    :param n: a positive integer
    :return an integer that is the largest prime number less than n
    """

    def is_prime(m: int) -> bool:
        if m < 2:
            return False
        if m in (2, 3):
            return True
        if m % 2 == 0 or m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True

    if n < 2:
        raise ValueError("n must be greater than 2")

    for j in range(n, 1, -1):
        if is_prime(j):
            return j
    return 2


def solve_modulo_equations(a: int, b: int, p: int, q: int) -> int:
    """Solve modulo equations
        x = a (mod p)
        x = b (mod q)
    where p and q are coprime.
    :param a: remainder a
    :param b: remainder b
    :param p: a prime number
    :param q: a prime number
    :return the unique solution x (mod p * q)
    """

    a = a % p
    b = b % q
    p_inv = pow(p, -1, q)
    q_inv = pow(q, -1, p)
    x = (a * q * q_inv + b * p * p_inv) % (p * q)
    return x

"""Wrapper for the derangement C library."""

from __future__ import annotations

import importlib.resources
from ctypes import CDLL, POINTER, c_char_p, c_int, create_string_buffer

lib_path = importlib.resources.files("permumark.derangement").joinpath("derangement.so")
lib = CDLL(str(lib_path))

lib.derangement_rank.restype = c_int
lib.derangement_rank.argtypes = [POINTER(c_int), c_int, c_char_p]
lib.derangement_unrank.restype = c_int
lib.derangement_unrank.argtypes = [c_char_p, c_int, POINTER(c_int)]


def large_int_to_string(num, chunk_size: int = 4000) -> str:
    """Convert a large integer to a string.
    :param num: a large integer
    :param chunk_size: size of chunk for conversion. Defaults to 4000
    :return a string representation of the large integer
    """
    if num == 0:
        return "0"

    chunks: list[str] = []
    base = 10**chunk_size

    while num > 0:
        num, r = divmod(num, base)
        chunks.append(str(r).zfill(chunk_size))

    chunks[-1] = chunks[-1].lstrip("0")
    return "".join(reversed(chunks))


def large_string_to_int(s: str, chunk_size: int = 4000) -> int:
    """Convert a string to a large integer.
    :param s: a string representation of a large integer
    :param chunk_size: size of chunk for conversion. Defaults to 4000
    :return the large integer
    """
    num = 0
    for i in range(0, len(s), chunk_size):
        num = num * (10 ** min(chunk_size, len(s) - i)) + int(s[i : i + chunk_size])
    return num


def derangement_unrank(k, n) -> list[int] | None:
    """Get the k-th derangement of n elements.
    :param k: the given rank
    :param n: the number of elements
    :return an n-element derangement of order k
    """
    perm_array = (c_int * n)()
    k_str = large_int_to_string(k)
    k_c_char_p = c_char_p(k_str.encode())
    status = lib.derangement_unrank(k_c_char_p, n, perm_array)
    if status == 0:
        buffer = memoryview(perm_array).cast("B").cast("i")
        return list(buffer[:n])
    return None


def derangement_rank(perm) -> int | None:
    """Get the rank of a derangement.
    :param perm: the given derangement
    :return the rank of the derangement
    """
    n = len(perm)
    perm_array = (c_int * n)(*perm)
    rank = create_string_buffer(10**5)
    status = lib.derangement_rank(perm_array, n, rank)
    if status == 0:
        return large_string_to_int(rank.value.decode())
    return None

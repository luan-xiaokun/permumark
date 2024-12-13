"""Permutation extraction based on linear sum assignment."""

from __future__ import annotations

from time import time
from typing import NamedTuple

import torch
from scipy.optimize import linear_sum_assignment


class PermExtractionResult(NamedTuple):
    """A named tuple to store the result of permutation extraction."""

    perm: list[int]
    eps_norm: float
    block_size: int | None
    block_perm: torch.Tensor | None
    time: float


def matrix_normalization(
    x: torch.Tensor, dim: str = "col", eps: float = 1e-8
) -> torch.Tensor:
    """Normalize a matrix into N(0, 1), either row-wise or column-wise."""
    dim = 1 if dim == "row" else 0
    return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True) + eps)


def extract_perm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    dim: str = "row",
    block_size: int | None = None,
    device: str | None = None,
) -> PermExtractionResult:
    """Solve equation B = P @ A + eps, where P is a permutation matrix and
    eps is a noise matrix, the goal is to minimize the Frobenius norm of eps.
    :param mat1: the original matrix, shape (m, n)
    :param mat2: the watermarked matrix, shape (m, n)
    :param dim: the dimension to permute. Defaults to "row"
    :param block_size: the block size to use. Defaults to None
    :param device: the device to use. Defaults to None, use cuda if available
    :return a tuple of permutation list and the Frobenius norm of eps
    """

    def get_eps_norm(
        m1: torch.Tensor, m2: torch.Tensor, p: list[int] | torch.Tensor
    ) -> float:
        return torch.norm(m2 - m1[p, :]).item() / torch.norm(m1).item()

    start_time = time()
    assert mat1.shape == mat2.shape, f"Different shapes {mat1.shape} and {mat2.shape}"

    # NOTE cdist does not support half precision
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mat1, mat2 = mat1.to(device).float(), mat2.to(device).float()

    # scale attack affects the second dimension (column, in torch; or row, in math)
    mat1 = matrix_normalization(mat1, dim="col")
    mat2 = matrix_normalization(mat2, dim="col")

    if dim == "col":
        mat1, mat2 = mat1.T, mat2.T
    m, d = mat1.shape

    # construct cost matrix for linear assignment problem
    if block_size is not None and (block_size < 0 or m % block_size != 0):
        raise ValueError(f"Invalid block size {block_size}")
    if block_size is None or block_size == 1:
        # switch to batch cdist if directly calling cdist costs more than 4GB RAM
        if m * m * d >= 1e12:
            cost_matrix = batch_cdist(mat1, mat2, batch_size=m // 4, p=2).cpu()
        else:
            cost_matrix = torch.cdist(mat1, mat2, p=2).cpu()
    else:
        cost_matrix = compute_block_perm_cost_matrix(mat1, mat2, block_size).cpu()
    # cost_matrix = cost_matrix + 1e-6 * torch.rand_like(cost_matrix)

    # solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    perm = torch.full((len(col_ind),), 0, dtype=torch.long)
    perm[col_ind] = torch.tensor(row_ind, dtype=torch.long)
    perm = perm.tolist()
    block_perm = None
    if block_size is not None and block_size > 1:
        block_perm = (
            torch.arange(block_size).repeat(len(perm))
            + torch.as_tensor(perm).repeat_interleave(block_size) * block_size
        )
        eps_norm = get_eps_norm(mat1, mat2, block_perm)
    else:
        eps_norm = get_eps_norm(mat1, mat2, perm)

    end_time = time()
    result = PermExtractionResult(
        perm=perm,
        eps_norm=eps_norm,
        block_size=block_size,
        block_perm=block_perm,
        time=end_time - start_time,
    )

    return result


def compute_block_perm_cost_matrix(
    mat1: torch.Tensor, mat2: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Given two matrices, compute the cost matrix for block matching.
    :param mat1: given matrix A
    :param mat2: given matrix B
    :param block_size: size of the block, e.g., head_dim=64
    :return a cost matrix of shape (block_num, block_num), where each cost is the
    average column-wise L2 distance between two blocks
    """
    assert mat1.shape == mat2.shape, f"Different shapes {mat1.shape} and {mat2.shape}"
    assert mat1.size(0) % block_size == 0, f"Got non-multiple row number {mat1.size(0)}"

    block_num = mat1.size(0) // block_size

    mat1_blocks = mat1.view(block_num, block_size, -1)
    mat2_blocks = mat2.view(block_num, block_size, -1)

    # cost_mat = torch.zeros(block_num, block_num, device=mat1.device)
    # for i in range(block_num):
    #     for j in range(block_num):
    #         cost_mat[i, j] = torch.sqrt(
    #             torch.sum((mat1_blocks[i] - mat2_blocks[j]) ** 2, dim=-1)
    #         ).mean()

    matching = (mat1_blocks[:, None, :, :] - mat2_blocks[None, :, :, :]) ** 2
    cost_mat = matching.sum(dim=-1).sqrt().mean(dim=-1)

    return cost_mat


def batch_cdist(
    mat1: torch.Tensor, mat2: torch.Tensor, batch_size: int, p: float = 2.0
) -> torch.Tensor:
    """
    Batch version of cdist to avoid OOM.
    :param mat1: first matrix of shape (n, d)
    :param mat2: second matrix of shape (m, d)
    :param batch_size: max batch size for cdist
    :param p: the order of the norm, default to L2 norm
    :return: a distance matrix of shape (n, m
    """
    n, _ = mat1.size()
    m, _ = mat2.size()

    result = torch.zeros((n, m), device=mat1.device)
    for i in range(0, n, batch_size):
        for j in range(0, m, batch_size):
            size1 = min(batch_size, n - i)
            size2 = min(batch_size, m - j)
            mat1_batch = mat1[i : i + size1]
            mat2_batch = mat2[j : j + size2]
            dist_batch = torch.cdist(mat1_batch, mat2_batch, p=p)
            result[i : i + size1, j : j + size2] = dist_batch

    return result

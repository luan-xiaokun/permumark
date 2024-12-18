from __future__ import annotations

import math

import torch
import tqdm
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Quantizer(nn.Module):
    def __init__(
        self,
        shape,
        bits: int,
        norm: float = 2.4,
        grid: int = 100,
        max_shrink: float = 0.8,
    ):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(2**bits - 1))
        self.register_buffer("minq", torch.zeros(shape))
        self.register_buffer("q", torch.zeros(shape))

        self.norm = norm
        self.grid = grid
        self.max_shrink = max_shrink

    def find_params(self, x: torch.Tensor):
        device = x.device
        shape = x.shape
        self.maxq = self.maxq.to(device)
        x = x.flatten(1)
        tmp = torch.zeros(x.shape[0], device=device)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = 1
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)
        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def ready(self) -> torch.Tensor:
        return torch.all(self.scale != 0)


class SparseGPT:
    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.device = layer.weight.device
        weight = layer.weight.data.clone()

        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Expected nn.Linear, got {type(layer)}")

        self.rows = weight.shape[0]
        self.cols = weight.shape[1]
        self.h_mat = torch.zeros((self.cols, self.cols), device=self.device)
        self.sample_num = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.h_mat *= self.sample_num / (self.sample_num + tmp)
        self.sample_num += tmp
        inp = math.sqrt(2 / self.sample_num) * inp.float()
        self.h_mat += inp.matmul(inp.t())

    @torch.no_grad()
    def fast_prune(
        self,
        sparsity: float,
        prune_n: int = 0,
        prune_m: int = 0,
        block_size: int = 128,
        perc_damp: float = 0.01,
    ):
        weight = self.layer.weight.data.detach().clone().float()

        if hasattr(self, "quantizer"):
            quantizer = getattr(self, "quantizer")
            if not quantizer.ready():
                self.quantizer.find_params(weight)

        h_mat = self.h_mat
        del self.h_mat
        dead = torch.diag(h_mat) == 0
        h_mat[dead, dead] = 1
        weight[:, dead] = 0

        losses = torch.zeros(self.rows, device=self.device)

        damp = perc_damp * torch.mean(torch.diag(h_mat))
        diag = torch.arange(self.cols, device=self.device)
        h_mat[diag, diag] += damp
        h_mat = torch.linalg.cholesky(h_mat)
        h_mat = torch.cholesky_inverse(h_mat)
        h_mat = torch.linalg.cholesky(h_mat, upper=True)
        h_inv = h_mat
        mask = None

        for i1 in range(0, self.cols, block_size):
            i2 = min(i1 + block_size, self.cols)
            count = i2 - i1

            weight1 = weight[:, i1:i2].clone()
            quant1 = torch.zeros_like(weight1)
            err1 = torch.zeros_like(weight1)
            losses1 = torch.zeros_like(weight1)
            h_inv1 = h_inv[i1:i2, i1:i2]

            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = weight1**2 / (torch.diag(h_inv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(weight1) == 1

            for i in range(count):
                w = weight1[:, i]
                d = h_inv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = (
                        weight1[:, i : i + prune_m] ** 2
                        / (torch.diag(h_inv1)[i : i + prune_m].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, "quantizer"):
                    q = quantize(
                        q.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    )

                quant1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2
                e = (w - q) / d
                weight1[:, i:] -= e.unsqueeze(1).matmul(h_inv1[i, i:].unsqueeze(0))
                err1[:, i] = e

            weight[:, i1:i2] = quant1
            losses += torch.sum(losses1, 1) / 2
            weight[:, i2:] -= err1.matmul(h_inv[i1:i2, i2:])

        torch.cuda.synchronize()

        self.layer.weight.data = weight.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.device
        )

    def free(self):
        self.h_mat = None
        torch.cuda.empty_cache()


def quantize(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int
) -> torch.Tensor:
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def find_layers(module: nn.Module, name="") -> dict[str, nn.Linear]:
    """
    Return a dictionary of all linear layers in the given module.
    :param module: given layer/module of a transformer model
    :param name: the name of the layer
    :return: a dictionary of all linear layers
    """
    if isinstance(module, nn.Linear):
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        rec_name = f"{name}.{child_name}" if name != "" else child_name
        res.update(find_layers(child, rec_name))
    return res


def prepare_dataloader(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int = 256
):
    def tokenize(example):
        inputs = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        dataset.map(tokenize, remove_columns=["text"], num_proc=8),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    return dataloader


def prepare_calibration_input(
    model: PreTrainedModel,
    dataloader: DataLoader,
    sample_num: int,
    device: str,
    max_length: int = 256,
):
    class Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    orig_dev = next(iter(model.parameters()))

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (sample_num, max_length, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(**batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module.to(orig_dev)
    model.model.embed_tokens = model.model.embed_tokens.to(orig_dev)
    model.model.norm = model.model.norm.to(orig_dev)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    return inps, outs, attention_mask, position_ids


def sparse_gpt_pruning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    sparsity: float,
    sample_num: int,
    prune_n: int,
    prune_m: int,
    block_size: int = 128,
    perc_damp: float = 0.01,
    device: str = "cuda",
    bits: int = 16,
    max_length: int = 256,
):
    dataloader = prepare_dataloader(
        dataset.select(range(sample_num)), tokenizer, max_length
    )

    model.to(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, sample_num, device, max_length
        )

    for i in tqdm.tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i].to(device)
        linear_layers = find_layers(layer)

        sparse_gpts = {}
        for name, linear_layer in linear_layers.items():
            sparse_gpts[name] = SparseGPT(linear_layer)
            if bits < 16:
                sparse_gpts[name].quantizer = Quantizer(1, bits)

        def add_batch(n: str):
            def add_batch_inner(_, inp, out):
                sparse_gpts[n].add_batch(inp[0].data, out.data)

            return add_batch_inner

        handles = []
        for name, linear_layer in linear_layers.items():
            handles.append(linear_layer.register_forward_hook(add_batch(name)))
        with torch.no_grad():
            for j in range(sample_num):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        for name in linear_layers.keys():
            sparse_gpts[name].fast_prune(
                sparsity, prune_n, prune_m, block_size, perc_damp
            )
            sparse_gpts[name].free()

        with torch.no_grad():
            for j in range(sample_num):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layers[i] = layer.cpu()
        del layer
        del sparse_gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

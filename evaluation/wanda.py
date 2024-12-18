from __future__ import annotations

import torch
import tqdm
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset

from sparse_gpt import find_layers, prepare_dataloader, prepare_calibration_input


class WrappedGPT:
    def __init__(self, layer, layer_id: int = 0, layer_name: str = "none") -> None:
        self.layer = layer
        self.device = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.cols = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros(self.cols, device=self.device)
        self.sample_num = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, _):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.sample_num / (self.sample_num + tmp)
        self.sample_num += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.sample_num


def wanda_pruning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    sparsity: float,
    sample_num: int,
    prune_n: int = 0,
    prune_m: int = 0,
    device: str | None = None,
    max_length: int = 256,
):

    dataloader = prepare_dataloader(
        dataset.select(range(sample_num)), tokenizer, max_length
    )

    model.to(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, sample_num, device, max_length
        )

    layers = model.model.layers
    for i in tqdm.tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i]
        linear_layers = find_layers(layer)

        wrapped_layers = {
            name: WrappedGPT(linear_layers[name]) for name in linear_layers
        }

        def add_batch(n):
            def add_batch_inner(_, inp, out):
                wrapped_layers[n].add_batch(inp[0].data, out[0].data)

            return add_batch_inner

        handles = []
        for name in wrapped_layers:
            handles.append(linear_layers[name].register_forward_hook(add_batch(name)))
        with torch.no_grad():
            for j in range(sample_num):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        for name, linear_layer in linear_layers.items():
            w_metric = torch.abs(linear_layer.weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            w_mask = torch.zeros_like(w_metric) == 1

            if prune_n != 0:
                for ii in range(w_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = w_metric[:, ii : ii + prune_m].float()
                        w_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(w_metric, dim=-1, stable=True)
                indices = sort_res[1][:, : int(w_metric.shape[1] * sparsity)]
                w_mask.scatter_(1, indices, True)

            linear_layer.weight.data[w_mask] = 0

        with torch.no_grad():
            for j in range(sample_num):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

"""Utility for transformer model modification and evaluation."""

from __future__ import annotations

import torch
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


def eval_predicted_tokens(
    model, model_path: str, dataset: Dataset, batch_size: int
) -> list[int]:
    """
    Evaluate the model by predicting the next token in the dataset.
    :param model: model to evaluate
    :param model_path: path of the model and its tokenizer
    :param dataset: dataset to evaluate
    :param batch_size: batch size for evaluation
    :return predicted tokens
    """

    def preprocess(examples):
        tokens = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=1024
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenizer_exceptions=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        if "Llama-3" in model_path:
            tokenizer.pad_token_id = 128004
        else:
            tokenizer.pad_token = tokenizer.eos_token

    processed_dataset = dataset.filter(
        lambda x: len(tokenizer.tokenize(x["text"])) >= 3
    ).map(preprocess, remove_columns=["text"])
    dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    predicted_next_token_ids = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            predicted_token_id = outputs.logits[:, -1].argmax(-1).tolist()
            predicted_next_token_ids.extend(predicted_token_id)

    model.to("cpu")

    return predicted_next_token_ids


def eval_perplexity(
    model, model_path, dataset: Dataset, batch_size: int = 128
) -> float:
    """
    Evaluate the model by calculating the perplexity.
    :param model: model to evaluate
    :param model_path: path of the model and its tokenizer
    :param dataset: dataset to evaluate
    :param batch_size: batch size for evaluation
    :return perplexity
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenizer_exceptions=False, trust_remote_code=True
    )
    texts = dataset["text"]
    input_ids_list = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        size = min(batch_size, len(texts) - i)
        encodings = tokenizer(texts[i : i + size], return_tensors="pt", padding=True)
        for ids in encodings["input_ids"]:
            input_ids_list.extend(ids[ids != tokenizer.pad_token_id].tolist())

    stride = 256
    seq_len = len(input_ids_list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlls = []

    model.eval()
    model.to(device)

    for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + 2 * stride, seq_len)
        input_ids = (
            torch.tensor(input_ids_list[begin_loc:end_loc]).unsqueeze(0).to(device)
        )
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()

from collections.abc import Mapping
from typing import List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


class TrainNLIAdapterDataLoader:
    def __init__(self, lang: str, pretrained_ck: str, max_length: Optional[int] = None):
        dataset = load_dataset("xnli", lang)
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_ck)
        self.max_length = max_length
        self.dataset = dataset.map(
            self.__tokenize_nli,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

    def __tokenize_nli(self, examples):
        tokenized_inputs = self.tokenizer(
            text=examples["premise"],
            text_pair=examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoded_inputs.items()}
        return batch

    def get_dataloader(self, batch_size: int = 16, types: List[str] = ["train", "validation"]):
        res = []
        for t in types:
            if t == "train":
                shuffle = True
            else:
                shuffle = False
            res.append(
                DataLoader(
                    self.dataset[t],
                    batch_size=batch_size,
                    collate_fn=self.__collate_fn,
                    num_workers=32,
                    shuffle=shuffle,
                )
            )
        return res

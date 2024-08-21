from typing import Dict, List, Optional

from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from UniBridge.MultiTok import SPTokenizerFast


class UniBridgeDataLoader:
    def __init__(
        self,
        lang: str,
        tok_pretrained_ck: str,
        prob: float,
        test_ratio: float,
        chunk_size: int = 384,
        max_length: Optional[int] = None,
        smart_chunk_size: bool = False,
        out_dict: Dict = {},
    ):
        dataset = load_dataset("multi_wiki", lang)["train"]
        self.tokenizer = SPTokenizerFast.from_pretrained(tok_pretrained_ck)
        self.max_length = max_length
        tokenized_datasets = dataset.map(
            self.__tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        if not smart_chunk_size:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = self.__analyze_chunk_size(tokenized_datasets)
        out_dict["chunk_size"] = self.chunk_size

        logger.info(f"Setting chunk size to {self.chunk_size} due to smart_chunk_size={smart_chunk_size}")
        dataset = tokenized_datasets.map(self.__group_texts, batched=True)
        dataset = dataset.train_test_split(test_size=test_ratio)
        self.dataset = dataset
        self.collator_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=prob)

    def __analyze_chunk_size(self, tokenized_datasets):
        len_data = tokenized_datasets.map(lambda example: {"len": len(example["input_ids"])})

        def avg(inp):
            return sum(inp) / len(inp)

        mean_seq_len = avg(len_data["len"])
        chunk_sizes = [128, 256, 384]
        for chunk_size in chunk_sizes:
            if mean_seq_len < chunk_size:
                return chunk_size
        return chunk_sizes[-1]

    def __group_texts(self, examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.chunk_size) * self.chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    def __tokenize_function(self, examples):
        result = self.tokenizer(examples["text"])
        return result

    def get_dataloader(self, batch_size: int = 16, types: List[str] = ["train", "test", "validation"]):
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
                    collate_fn=self.collator_fn,
                    num_workers=32,
                    shuffle=shuffle,
                )
            )
        return res

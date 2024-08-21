from typing import List, Optional

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, XLMRobertaTokenizerFast


class MLMAdapterDataLoader:
    def __init__(
        self,
        lang: str,
        model: str,
        pretrained_ck: str,
        prob: float,
        test_ratio: float,
        chunk_size: int = 256,
        max_length: Optional[int] = None,
    ):
        dataset = load_dataset("multi_wiki", lang)
        if model in ["mbert", "mbert_cased"]:
            self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_ck)
        elif model == "xlm-r":
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_ck)
        else:
            assert f"parameter `model` for MLMAdapterDataLoader must be either ['mbert', 'xlm-r', 'mbert_cased'], {model} does not belong to that"
        self.max_length = max_length
        self.chunk_size = chunk_size
        tokenized_datasets = dataset.map(
            self.__tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        dataset = tokenized_datasets.map(self.__group_texts, batched=True)
        dataset = dataset["train"].train_test_split(test_size=test_ratio)
        self.dataset = dataset
        self.collator_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=prob)

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

    def __tokenize(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

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

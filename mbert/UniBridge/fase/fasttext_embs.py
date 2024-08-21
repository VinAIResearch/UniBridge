import multiprocessing
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import fasttext
from datasets import Dataset
from datasets.fingerprint import Hasher
from datasets.load import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from .download_utils import download, gunzip


CACHE_DIR = (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "fase").expanduser().resolve()


def train_fasttext(
    text_path: str,
    target_tokenizer: PreTrainedTokenizerFast,
    epochs,
    dim,
    processes=None,
    cache_tokenized_text=False,
):
    processes = processes or multiprocessing.cpu_count()
    target_tokenizer_hash = Hasher().hash(target_tokenizer)
    data_file_name = Path(text_path).stem
    data_file_suffix = Path(text_path).suffix

    print(target_tokenizer_hash, data_file_name, data_file_suffix)
    cache_file = CACHE_DIR / "data" / f"{data_file_name}_tokenized_{target_tokenizer_hash}{data_file_suffix}"

    if cache_file.exists():
        logger.success(f"Tokenized text for {text_path} found at {cache_file}...")
    else:
        if cache_tokenized_text:
            logger.info(f"Tokenizing text in {text_path} and caching results in {cache_file}...")

        if text_path.endswith(".txt"):
            data = []
            with open(text_path, "r") as fin:
                for line in fin:
                    data.append({"text": re.sub("\n", "", line)})
            dataset = Dataset.from_list(data)
        if text_path.endswith(".json") or text_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=text_path, split="train")
        dataset = dataset.map(
            lambda sample: {"text": " ".join([token for token in target_tokenizer.tokenize(sample["text"])])},
            num_proc=8,
        )
        if cache_tokenized_text:
            os.makedirs(str(cache_file.parent), exist_ok=True)

            with cache_file.open("w+", encoding="utf-8") as f:
                f.writelines((text + "\n" for text in tqdm(dataset["text"], desc="Writing data...")))
            logger.success(f"Tokenized target language training data for fasttext written to {cache_file}...")
        else:
            temp_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8")
            for text in dataset["text"]:
                temp_file.write(text + "\n")
            cache_file = temp_file.name

    logger.info(f"Training fasttext model on {f'tokenized {text_path}' if cache_tokenized_text else cache_file}...")
    # We use CBOW instead of skipgram becasue CBOW is more closely aligned with Masked Language Modeling
    return fasttext.train_unsupervised(
        str(cache_file),
        dim=dim,
        neg=10,
        model="cbow",
        epoch=epochs,
        thread=processes,
        minCount=1,
    )


def download_pretrained_fasttext_word_embs(identifier: str, verbose=True):
    """
    Utility function to download and cache embeddings from https://fasttext.cc.
    Args:
        identifier: 2-letter language code.
    Returns:
        fastText model loaded from https://fasttext.cc/docs/en/crawl-vectors.html.
    ------
    From https://github.com/CPJKU/wechsel/blob/56ae305e5d7d20383cf891371ffeb7885763cdc5/wechsel/__init__.py#L184-L211
    """
    if os.path.exists(identifier):
        path = Path(identifier)
    else:
        logger.info(
            f"Loading fasttext *word* embeddings for language '{identifier}' from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{identifier}.300.bin.gz."
        )
        path = CACHE_DIR / "pretrained_fasttext" / f"cc.{identifier}.300.bin"

        if not path.exists():
            path = download(
                f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{identifier}.300.bin.gz",
                CACHE_DIR / "pretrained_fasttext" / f"cc.{identifier}.300.bin.gz",
                verbose=verbose,
            )
            path = gunzip(path)
        else:
            logger.success(f"Found cache at {path}.")

    return fasttext.load_model(str(path))


def load_target_token_embedding(
    *,
    target_tokenizer: PreTrainedTokenizerFast,
    target_training_data_path: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
    language_identifier: Optional[str] = None,
    fasttext_epochs: Optional[int] = None,
    fasttext_embedding_dim: Optional[int] = None,
    processes: Optional[int] = None,
):
    """Load fasttext token embeddings necessary for FASE. You have three choices:
    1. (Preferred) Provide `target_tokenizer` and `target_training_data_path`. The function will then tokenize the training data and train a fasttext model on it. This is the method from the FASE paper. `target_training_data_path` is expected to point to a file containing lines of text in the target language.
    2. Provide `target_tokenizer` and the language identifier of the target language. The function will then download pretrained fasttext *word* embeddings for the target language and embed the tokens of the target tokenizer into the pretrained fasttext word embedding space. This is akin to the method used in the WECHSEL paper.
    3. Provide a path to a pretrained fasttext model with embeddings for tokens in your `target_tokenizer`'s vocab.
    """
    if target_tokenizer and target_training_data_path:
        args = {}
        if fasttext_epochs:
            args["epochs"] = fasttext_epochs
        if fasttext_embedding_dim:
            args["dim"] = fasttext_embedding_dim
        fasttext_model = train_or_load_fasttext_model(
            target_training_data_path,
            target_tokenizer,
            model_cache_path=fasttext_model_path,
            processes=processes,
            **args,
        )
    elif fasttext_model_path:
        fasttext_model = fasttext.load_model(fasttext_model_path)
    elif target_tokenizer and language_identifier:
        # Embed tokens into pretrained fasttext word embedding space like in WECHSEL (Minixhofer et al., https://github.com/CPJKU/wechsel/)
        fasttext_model = download_pretrained_fasttext_word_embs(language_identifier)
        fasttext_token_embs = {}
        for token, idx in target_tokenizer.get_vocab().items():
            clean_token = target_tokenizer.decode(idx).strip()
            fasttext_token_embs[token] = fasttext_model.get_word_vector(clean_token)

        # NOTE: This is fine, since we only use the fasttext model as a dict from tokens to embeddings.
        fasttext_model = fasttext_token_embs
    else:
        raise ValueError(
            "You must provide either a tokenizer and training data in the target language, a language identifier or a path to a fasttext model."
        )
    return fasttext_model


def train_or_load_fasttext_model(
    text_path: Optional[str],
    target_tokenizer: PreTrainedTokenizerFast,
    epochs=3,
    dim=300,
    model_cache_path=None,
    processes=None,
):
    data_file_name = Path(text_path).stem
    data_file_suffix = Path(text_path).suffix
    target_tokenizer_hash = Hasher().hash(target_tokenizer)

    model_cache_path = Path(
        model_cache_path
        or (
            CACHE_DIR
            / "fasttext"
            / f"data_{data_file_name}_{data_file_suffix}_tokenizer_{target_tokenizer_hash}_epochs_{epochs}_dim_{dim}.bin"
        )
    )

    if not model_cache_path.exists():
        fasttext_model = train_fasttext(text_path, target_tokenizer, epochs=epochs, dim=dim, processes=processes)
        logger.info(f"Saving fasttext model to {model_cache_path}.")
        os.makedirs(str(model_cache_path.parent), exist_ok=True)
        fasttext_model.save_model(str(model_cache_path))
    else:
        logger.info(f"Loading pretrained fasttext model from {model_cache_path}.")
        fasttext_model = fasttext.load_model(str(model_cache_path))
    return fasttext_model

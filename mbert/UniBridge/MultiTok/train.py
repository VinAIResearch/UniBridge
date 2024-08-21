import json
import math
import os
import re
from typing import List

import sentencepiece as spm
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


def avg(ls: List):
    return sum(ls) / len(ls)


class Tokenizer(object):
    def __init__(self, vocab_file):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))

    def get_vocab(self):
        return [self.sp_model.IdToPiece(idx) for idx in range(len(self.sp_model))]

    def tokenize(self, text):
        return self.sp_model.EncodeAsIds(text)


def compute_alp(input_file, vocab_file):
    with open(input_file, "r") as fin:
        lines = fin.readlines()

    all_tokens = 0
    tokenizer = Tokenizer(vocab_file)
    words_list = tokenizer.get_vocab()
    words = {}
    for i, word in enumerate(words_list):
        words[i] = 0

    tokenized_lines = []
    for line in tqdm(lines):
        line = line.strip()
        token_ids = tokenizer.tokenize(line)
        all_tokens += len(token_ids)
        for idx in token_ids:
            words[idx] += 1
        tokenized_lines.append(token_ids)
    for idx in words.keys():
        words[idx] /= all_tokens
    probs = []
    for token_ids in tokenized_lines:
        p = 0.0
        for idx in token_ids:
            p += math.log(words[idx])
        probs.append(p)

    return avg(probs)


def get_stop_punct(lang: str) -> str:
    if lang in ["lij", "en"]:
        return ".!?"
    return "."


def split_paragraph(line: str, lang: str) -> List[str]:
    puncts = get_stop_punct(lang)
    default_punct = puncts[0]
    sent = ""

    line = re.sub("(\d)\.(\d)", r"\1,\2", line)

    sents = []
    line = list(line)
    if line[-1] not in list(puncts):
        line.append(default_punct)
    for char in line:
        sent += char
        if char in puncts:
            sents.append(sent)
            sent = ""

    sents = [sent for sent in sents if len(sent) > 0]
    return sents


def train(
    lang: str,
    data_dir: str,
    tokenizer_dir: str,
    use_alp: bool = False,
    min_vocab: int = 10_000,
    max_vocab: int = 100_000,
    vocab_step: int = 1000,
    alp_threshold: float = 5.0,
    debug: bool = False,
    is_bpe: bool = True,
):
    logger.info(f"Loading {lang} dataset ...")
    dataset = load_dataset("multi_wiki", lang)["train"]

    logger.info("Writing text data to file ...")
    data = dataset["text"]
    os.makedirs(f"{data_dir}/{lang}", exist_ok=True)
    with open(f"{data_dir}/{lang}/{lang}.txt", "w") as fout:
        for line in data:
            if len(line) > 0:
                fout.write(line + "\n")
    if is_bpe:
        vocab_build_algo = "bpe"
    else:
        vocab_build_algo = "unigram"
    logger.info(f"Building tokenizer with alp={use_alp} and min/max vocab={min_vocab}/{max_vocab} ...")
    if lang in "zh ja ko wuu".split(" "):
        c_cov = 0.9995
    else:
        c_cov = 1.0
    if not use_alp:
        spm.SentencePieceTrainer.train(
            input=f"{data_dir}/{lang}/{lang}.txt",
            model_prefix=f"sentencepiece.{vocab_build_algo}",
            vocab_size=min_vocab,
            model_type=vocab_build_algo,
            character_coverage=c_cov,
        )
    else:
        if debug:
            alp_vocabs = []
        previous_alp = None
        vocab_peak = False
        for vocab_size in range(min_vocab, max_vocab + 1, vocab_step):
            try:
                spm.SentencePieceTrainer.train(
                    input=f"{data_dir}/{lang}/{lang}.txt",
                    model_prefix=f"sentencepiece.{vocab_build_algo}",
                    vocab_size=vocab_size,
                    model_type=vocab_build_algo,
                    character_coverage=c_cov,
                )
            except Exception:
                vocab_peak = True
                vocab_size -= vocab_step
                spm.SentencePieceTrainer.train(
                    input=f"{data_dir}/{lang}/{lang}.txt",
                    model_prefix=f"sentencepiece.{vocab_build_algo}",
                    vocab_size=vocab_size,
                    model_type=vocab_build_algo,
                    character_coverage=c_cov,
                )
            logger.info(f"Calculating ALP with vocab size={vocab_size}...")
            alp = compute_alp(
                input_file=f"{data_dir}/{lang}/{lang}.txt", vocab_file=f"sentencepiece.{vocab_build_algo}.model"
            )
            if debug:
                alp_vocabs.append(
                    {
                        "vocab_size": vocab_size,
                        "alp": alp,
                        "diff": 0.0 if previous_alp is None else abs(alp - previous_alp),
                    }
                )
            if vocab_peak:
                break
            if previous_alp is not None:
                if abs(alp - previous_alp) < alp_threshold:
                    break
                    # pass
                else:
                    previous_alp = alp
            else:
                previous_alp = alp

        if debug:
            with open(f"study_alp_{lang}.json", "w") as fout:
                json.dump(alp_vocabs, fout)
    path = f"{tokenizer_dir}/spm/{lang}"
    os.makedirs(path, exist_ok=True)
    os.rename(f"sentencepiece.{vocab_build_algo}.model", f"{path}/sentencepiece.{vocab_build_algo}.model")
    os.remove(f"sentencepiece.{vocab_build_algo}.vocab")

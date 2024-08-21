import os

import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from UniBridge.fase import FASE
from UniBridge.MultiTok import SPTokenizerFast


def build_tokenizer(tokenizer_ckpt: str, data_path: str, src_tokenizer: str, outdir: str):
    source_tokenizer = BertTokenizerFast.from_pretrained(src_tokenizer)
    source_model = BertForMaskedLM.from_pretrained(src_tokenizer)

    target_tokenizer = SPTokenizerFast.from_pretrained(tokenizer_ckpt)
    target_embeddings = FASE(
        source_embeddings=source_model.get_input_embeddings().weight,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        target_training_data_path=data_path,
    )
    os.makedirs(outdir, exist_ok=True)
    torch.save(target_embeddings, f"{outdir}/embedding.pt")

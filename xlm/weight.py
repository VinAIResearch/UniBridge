import json
import os
import re
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import XLMRobertaAdapterModel, XLMRobertaTokenizerFast
from UniBridge.Embedding import UniBridgeEmbedding
from UniBridge.MultiTok import SPTokenizerFast


def flores_mapping(lang: str) -> str:
    mappings = {
        # source language
        "en": "eng_Latn",
        "ru": "rus_Cyrl",
        "ja": "jpn_Jpan",
        "zh": "zho_Hant",
        "ar": "arb_Arab",
        # target language
        "am": "amh_Ethi",
        "ang": "ang_Latn",
        "aym": "aym_Latn",
        "bzd": "bzd_Latn",
        "cdo": "cdo_Latn",
        "cni": "cni_Latn",
        "crh": "crh_Latn",
        "eml": "eml_Latn",
        "frr": "frr_Latn",
        "gn": "grn_Latn",
        "hch": "hch_Latn",
        "km": "khm_Khmr",
        "kn": "kan_Knda",
        "lij": "lij_Latn",
        "nah": "nah_Latn",
        "olo": "olo_Latn",
        "oto": "oto_Latn",
        "ps": "pbt_Arab",
        "quy": "quy_Latn",
        "sa": "san_Deva",
        "sd": "snd_Arab",
        "shp": "shp_Latn",
        "si": "sin_Sinh",
        "so": "som_Latn",
        "ta": "tam_Taml",
        "tar": "tar_Latn",
        "tl": "tgl_Latn",
        "tt": "tat_Cyrl",
    }
    return mappings[lang]


def get_path(src_lang: str, tgt_lang: str, from_flores: bool = True) -> str:
    if from_flores:
        path = f"alignment/flores/{tgt_lang}"
    else:
        path = f"alignment/opus/{tgt_lang}"
    src_lang = flores_mapping(src_lang)
    tgt_lang = flores_mapping(tgt_lang)
    return f"{path}/{tgt_lang}-{src_lang}.txt", len("eng_Latn: ")


def load_data(path: str, is_src: bool, prefix_len: int) -> List[str]:
    with open(path, "r") as fin:
        lines = fin.readlines()

    lines = list(map(lambda x: re.sub("\n", "", x), lines))
    lines = list(filter(lambda x: len(x) > 0, lines))
    texts = []
    for idx, line in enumerate(lines):
        if idx % 2 == 0 and is_src:
            text = line[prefix_len:]
            texts.append(text)
        if idx % 2 == 1 and not is_src:
            text = line[prefix_len:]
            texts.append(text)
    return texts


def mean_pooling(model_output: Dict, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def gen_weight(tgt_lang: str, from_flores: bool = True):
    ####################
    #                  #
    # SOURCE LANGUAGES #
    #                  #
    ####################
    model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    src_langs = ["en", "ru", "zh", "ar"]
    src_tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    tgt_langs = [tgt_lang]

    # load adapters
    for lang in tqdm(src_langs):
        model.load_adapter(f"src_ckpt/lang/{lang}/adapter", load_as=lang, leave_out=[11])  # src_lang

    outs = []
    model.to(device)

    for lang in tqdm(src_langs):
        model.set_active_adapters([f"{lang}"])

        path, prefix_len = get_path(lang, tgt_langs[0], from_flores=from_flores)
        data = load_data(path, is_src=True, prefix_len=prefix_len)
        batch = src_tokenizer(data, return_tensors="pt", padding=True)

        model.eval()
        with torch.inference_mode():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            out = mean_pooling(out, batch["attention_mask"])
            outs.append(out)
    del model

    ####################
    #                  #
    # TARGET LANGUAGES #
    #                  #
    ####################
    tgt_lang = tgt_langs[0]
    model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
    tgt_tokenizer = SPTokenizerFast.from_pretrained(f"tokenizer_ckpt/sptok/{tgt_lang}")

    # replace embedding
    embed_pretrained_ckpt = f"tgt_ckpt/lang/{tgt_lang}/embedding"
    model.set_input_embeddings(UniBridgeEmbedding.from_pretrained(embed_pretrained_ckpt))
    # load adapter
    model.load_adapter(f"tgt_ckpt/lang/{tgt_lang}/adapter", load_as=tgt_lang, leave_out=[11])  # tgt_lang
    model.to(device)
    model.set_active_adapters([f"{tgt_lang}"])

    # load data
    path, prefix_len = get_path(src_langs[0], tgt_lang, from_flores=from_flores)
    data = load_data(path, is_src=False, prefix_len=prefix_len)
    batch = tgt_tokenizer(data, return_tensors="pt", padding=True)

    tgt_outs = []
    model.eval()
    with torch.inference_mode():
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        out = mean_pooling(out, batch["attention_mask"])
        tgt_outs.append(out)
    del model

    ##############
    #            #
    # DIFFERENCE #
    #            #
    ##############

    diffs = []

    def l2_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(torch.pow(torch.subtract(a, b), 2), dim=-1))

    for out in outs:
        distance = torch.mean(l2_dist(out, tgt_outs[0]))
        diffs.append(distance)

    diffs = torch.stack(diffs)
    sim: List[float] = F.softmax(torch.divide(1, diffs), dim=0).cpu().tolist()

    result = {}
    result.update(zip(src_langs, sim))
    os.makedirs("harmony_weights", exist_ok=True)
    try:
        with open("harmony_weights/weights.json", "r") as fin:
            data = json.load(fin)
    except Exception:
        data = {}
    data[tgt_langs[0]] = result
    with open("harmony_weights/weights.json", "w") as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    mappings = [
        {"tgt_lang": "am", "from_flores": True},
        {"tgt_lang": "ang", "from_flores": False},
        {"tgt_lang": "aym", "from_flores": False},
        {"tgt_lang": "bzd", "from_flores": False},
        {"tgt_lang": "cdo", "from_flores": False},
        {"tgt_lang": "cni", "from_flores": False},
        {"tgt_lang": "crh", "from_flores": True},
        {"tgt_lang": "eml", "from_flores": False},
        {"tgt_lang": "frr", "from_flores": False},
        {"tgt_lang": "gn", "from_flores": True},
        {"tgt_lang": "hch", "from_flores": False},
        {"tgt_lang": "km", "from_flores": True},
        {"tgt_lang": "kn", "from_flores": True},
        {"tgt_lang": "lij", "from_flores": True},
        {"tgt_lang": "nah", "from_flores": False},
        {"tgt_lang": "olo", "from_flores": False},
        {"tgt_lang": "oto", "from_flores": False},
        {"tgt_lang": "ps", "from_flores": True},
        {"tgt_lang": "quy", "from_flores": True},
        {"tgt_lang": "sa", "from_flores": True},
        {"tgt_lang": "sd", "from_flores": True},
        {"tgt_lang": "shp", "from_flores": False},
        {"tgt_lang": "si", "from_flores": True},
        {"tgt_lang": "so", "from_flores": True},
        {"tgt_lang": "ta", "from_flores": True},
        {"tgt_lang": "tar", "from_flores": False},
        {"tgt_lang": "tl", "from_flores": True},
        {"tgt_lang": "tt", "from_flores": True},
    ]
    for mapping in mappings:
        gen_weight(**mapping)

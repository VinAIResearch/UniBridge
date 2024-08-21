import argparse
import os
import shutil

from UniBridge.Embedding import UniBridgeEmbedConfig, UniBridgeEmbedding, build_tokenizer
from UniBridge.MultiTok import convert_tok, train


def get_args():
    parser = argparse.ArgumentParser(description="Get the language")
    parser.add_argument("--lang", type=str, help="Langauge to UniBridge", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    lang = args.lang
    train(
        lang,
        "monolingual_data",
        "tokenizer_ckpt",
        use_alp=True,
        min_vocab=3_000,
        max_vocab=60_000,
        vocab_step=1_000,
        is_bpe=False,
    )  # , debug=True)
    convert_tok(lang, "tokenizer_ckpt", is_bpe=False)
    build_tokenizer(
        f"tokenizer_ckpt/sptok/{lang}",
        f"monolingual_data/{lang}/{lang}.txt",
        "bert-base-multilingual-cased",
        f"init_embed_ckpt/{lang}",
    )
    unused_data_dir = f"monolingual_data/{lang}"
    shutil.rmtree(unused_data_dir)
    config = UniBridgeEmbedConfig(pretrain_ckpt=f"init_embed_ckpt/{lang}/embedding.pt")
    model = UniBridgeEmbedding(config)
    model.save_pretrained(f"init_embed_ckpt/{lang}/embed_model", safe_serialization=False)
    os.remove(f"init_embed_ckpt/{lang}/embedding.pt")
    path = f"{os.path.expanduser('~')}/.cache/fase"
    shutil.rmtree(path)

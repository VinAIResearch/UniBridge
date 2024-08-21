import shutil

from UniBridge.MultiTok import SPTokenizer, SPTokenizerFast, convert_slow_tokenizer


def convert_tok(lang: str, tokenizer_dir: str, is_bpe: bool = True):
    print("Initialize sentencepiece tokenizer from checkpoint ...")
    if is_bpe:
        vocab_build_algo = "bpe"
    else:
        vocab_build_algo = "unigram"
    tokenizer = SPTokenizer(f"{tokenizer_dir}/spm/{lang}/sentencepiece.{vocab_build_algo}.model")
    fast_tokenizer = convert_slow_tokenizer(tokenizer)
    wrapper_tokenizer = SPTokenizerFast(tokenizer_object=fast_tokenizer)
    out = wrapper_tokenizer.encode("this is a test")
    print(out)
    text1 = wrapper_tokenizer.decode(out, skip_special_tokens=True)
    print(text1)
    wrapper_tokenizer.save_pretrained(f"{tokenizer_dir}/sptok/{lang}")
    old_tokenizer_dir = f"{tokenizer_dir}/spm/{lang}"
    shutil.rmtree(old_tokenizer_dir)

import json
import os

import evaluate
import pandas as pd
import torch
from tqdm import tqdm
from transformers import XLMRobertaAdapterModel
from UniBridge import PosAncDataLoader, PosLabelConverter, UniBridgeEmbedding


target_langs = (
    "am_att lij_glt olo_kkpp sa_ufal+sa_vedic sd_mazhardootio si_stb ta_mwtt tl_trg+tl_ugnayan tt_nmctt".split(" ")
)
src_langs = ["en", "zh", "ja", "ar", "ru"]
model_type = "xlm-r"  # mbert, xlm-r
tokenizer_ckpt = "xlm-roberta-base"  # bert-base-multilingual-cased, xlm-roberta-base
max_length = 510


def postprocess(lc, predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[lc.id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [lc.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels


def infer(tgt_lang: str, model_type: str, tokenizer_ckpt: str, max_length: int):
    _tgt_lang = tgt_lang
    tgt_lang = _tgt_lang.split("_")[0]
    target_lang = _tgt_lang

    # loading harmony weights
    with open("harmony_weights/weights.json", "r") as fin:
        data = json.load(fin)
    lang_weights = data[tgt_lang]

    # loading model
    print(f"Loading model with target lang={tgt_lang}, model type={model_type}")
    model = XLMRobertaAdapterModel.from_pretrained(tokenizer_ckpt)

    # load lang adapter
    print("Loading language adapter ...")
    model.set_input_embeddings(UniBridgeEmbedding.from_pretrained(f"tgt_ckpt/lang/{tgt_lang}/embedding"))
    model.load_adapter(f"tgt_ckpt/lang/{tgt_lang}/adapter", load_as=tgt_lang, leave_out=[11])

    # load task adapter
    parallel_blk = []
    for lang in tqdm(src_langs):
        model.load_adapter(f"src_ckpt/pos/{lang}", load_as=f"{lang}_pos", leave_out=[11])
        parallel_blk.append(f"{lang}_pos")

    device = torch.device("cuda:0")
    model = model.to(device)

    lc = PosLabelConverter()

    # loading dataloader
    hyperparameter = {
        "pretrained_ck": f"tokenizer_ckpt/sptok/{tgt_lang}",
        "ls_sub_data": target_lang,
        "is_target_lang": True,
        "max_length": max_length,
    }
    pos_dataloader = PosAncDataLoader(**hyperparameter)
    [test_dataloader] = pos_dataloader.get_dataloader(batch_size=32, types=["test"])

    # metric
    test_metric = evaluate.load("metrics/accuracy.py")

    # inference
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.inference_mode():
            final_logits = None
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            for lang in src_langs:
                model.set_active_adapters([tgt_lang, f"{lang}_pos"])
                logits = model(**batch)["logits"]
                if final_logits is not None:
                    final_logits += logits * float(lang_weights[lang])
                else:
                    final_logits = logits * float(lang_weights[lang])

            predictions = final_logits.argmax(dim=-1)

        decoded_preds, decoded_labels = postprocess(lc, predictions, labels)
        test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = test_metric.compute()
    accuracy = results["accuracy"] * 100
    return accuracy


if __name__ == "__main__":
    accuracies = []
    for lang in target_langs:
        accuracy = infer(lang, model_type, tokenizer_ckpt, max_length)
        accuracies.append(accuracy)
    df = pd.DataFrame({"lang": target_langs, "acc": accuracies})
    os.makedirs("result", exist_ok=True)
    df.to_csv(f"result/pos-{model_type}.csv", index=False)

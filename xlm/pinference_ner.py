import json
import os

import evaluate
import pandas as pd
import torch
from tqdm import tqdm
from transformers import XLMRobertaAdapterModel
from UniBridge import NerLabelConverter, TagAncDataLoader, UniBridgeEmbedding


target_langs = ["am", "ang", "cdo", "crh", "eml", "frr", "km", "kn", "lij", "ps", "sa", "sd", "si", "so"]
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
    return true_labels, true_predictions


def infer(tgt_lang: str, model_type: str, tokenizer_ckpt: str, max_length: int):
    # loading harmony weights
    with open("harmony_weights/weights.json", "r") as fin:
        data = json.load(fin)
    lang_weights = data[tgt_lang]

    # loading model
    print(f"Loading model with target lang={tgt_lang}, model type={model_type}")

    model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
    # load lang adapter
    print("Loading language adapter ...")
    model.set_input_embeddings(UniBridgeEmbedding.from_pretrained(f"tgt_ckpt/lang/{tgt_lang}/embedding"))
    model.load_adapter(f"tgt_ckpt/lang/{tgt_lang}/adapter", load_as=tgt_lang, leave_out=[11])

    # load task adapter
    parallel_blk = []
    for lang in tqdm(src_langs):
        model.load_adapter(f"src_ckpt/ner/{lang}", load_as=f"{lang}_ner", leave_out=[11])
        parallel_blk.append(f"{lang}_ner")

    device = torch.device("cuda:0")
    model = model.to(device)

    lc = NerLabelConverter()

    # loading dataloader
    ner_dataloader = TagAncDataLoader(
        lang=tgt_lang, model=model_type, tok_pretrained_ck=f"tokenizer_ckpt/sptok/{tgt_lang}", max_length=max_length
    )
    [test_dataloader] = ner_dataloader.get_dataloader(batch_size=32, types=["test"])

    # metric
    test_metric = evaluate.load("metrics/seqeval.py")

    # inference
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.inference_mode():
            final_logits = None
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            for lang in src_langs:
                model.set_active_adapters([tgt_lang, f"{lang}_ner"])
                logits = model(**batch)["logits"]
                if final_logits is not None:
                    final_logits += logits * float(lang_weights[lang])
                else:
                    final_logits = logits * float(lang_weights[lang])

            predictions = final_logits.argmax(dim=-1)

        decoded_preds, decoded_labels = postprocess(lc, predictions, labels)
        test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = test_metric.compute()
    f1 = results["overall_f1"] * 100
    return f1


if __name__ == "__main__":
    f1s = []
    for lang in target_langs:
        f1 = infer(lang, model_type, tokenizer_ckpt, max_length)
        f1s.append(f1)
    df = pd.DataFrame({"lang": target_langs, "f1": f1s})
    os.makedirs("result", exist_ok=True)
    df.to_csv(f"result/ner-{model_type}.csv", index=False)

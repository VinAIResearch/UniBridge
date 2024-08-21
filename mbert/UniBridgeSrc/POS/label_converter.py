class LabelConverter:
    def __init__(self):
        self.id2label = [
            "NOUN",
            "PUNCT",
            "ADP",
            "NUM",
            "SYM",
            "SCONJ",
            "ADJ",
            "PART",
            "DET",
            "CCONJ",
            "PROPN",
            "PRON",
            "X",
            "_",
            "ADV",
            "INTJ",
            "VERB",
            "AUX",
        ]
        self.label2id = {v: idx for idx, v in enumerate(self.id2label)}

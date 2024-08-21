class LabelConverter:
    def __init__(self):
        self.id2label = ["entailment", "neutral", "contradiction"]
        self.label2id = {v: idx for idx, v in enumerate(self.id2label)}

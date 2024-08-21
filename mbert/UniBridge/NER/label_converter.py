class LabelConverter:
    def __init__(self):
        self.label2id = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
        }
        self.id2label = [k for k in self.label2id.keys()]
        self.abbr2full = {
            "PER": "Person",
            "ORG": "Organization",
            "LOC": "Location",
        }

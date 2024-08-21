import json
import os

import datasets

from .langdef import LANGUAGES


class MultiWikiConfig(datasets.BuilderConfig):
    def __init__(self, language, version, **kwargs):
        super().__init__(
            name=language,
            description=f"Filtered Wikipedia dataset for {language}.",
            version=version,
            **kwargs,
        )


class MultiWiki(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = MultiWikiConfig
    # BUILDER_CONFIGS = [datasets.BuilderConfig(name='scn', version=VERSION, description='scn')]
    BUILDER_CONFIGS = [MultiWikiConfig(language=lang, version=datasets.Version("1.0.0")) for lang in LANGUAGES]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="",
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage="",
            # License for the dataset if available
            license="",
            # Citation for the dataset
            citation="",
        )

    def _split_generators(self, dl_manager):
        data_dir = f"data/{self.config.name}/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "text_wiki.jsonl"),
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                # Yields examples as (key, example) tuples
                yield key, {
                    "text": data["text"],
                }

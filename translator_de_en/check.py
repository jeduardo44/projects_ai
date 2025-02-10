"""
from datasets import get_dataset_config_names
print(get_dataset_config_names("wmt16"))

from datasets import load_dataset
dataset = load_dataset("wmt16", "de-en")
print(dataset["train"].column_names)
"""
from datasets import load_dataset

dataset = load_dataset("wmt16", "de-en")
print(dataset["train"][0])
print(dataset["train"][0]["translation"]) 
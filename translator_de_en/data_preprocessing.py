from datasets import load_dataset
from transformers import T5Tokenizer
from constants import *

def preprocess_function(examples, tokenizer):
    src_texts = [ex[SOURCE_LANG] for ex in examples["translation"]]
    tgt_texts = [ex[TARGET_LANG] for ex in examples["translation"]]

    inputs = tokenizer(src_texts, padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(tgt_texts, padding="max_length", truncation=True, max_length=128)

    inputs["labels"] = targets["input_ids"]
    return inputs

def get_small_dataset(tokenizer, sample_size=500):
    dataset = load_dataset(DATASET_NAME, f"{SOURCE_LANG}-{TARGET_LANG}")

    # Pegar uma amostra pequena do dataset
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(sample_size))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(sample_size // 10))

    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    return tokenized_dataset

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    dataset = get_small_dataset(tokenizer)
    print("Dataset reduzido carregado com sucesso!")

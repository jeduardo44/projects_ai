from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from constants import *
from dataset_preprocessing import get_small_dataset

# Carregar tokenizer e modelo
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Carregar dataset já processado
dataset = get_small_dataset(tokenizer)

# Configuração de treinamento
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,  
    logging_dir="./logs",
    save_total_limit=1,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Treinamento finalizado e modelo salvo em:", SAVE_DIR)

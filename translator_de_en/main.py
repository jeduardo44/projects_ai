from fastapi import FastAPI
from transformers import T5ForConditionalGeneration, T5Tokenizer
from constants import SAVE_DIR

app = FastAPI()

# Carregar modelo treinado
tokenizer = T5Tokenizer.from_pretrained(SAVE_DIR)
model = T5ForConditionalGeneration.from_pretrained(SAVE_DIR)

@app.get("/translate")
def translate(text: str):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"translation": translation}

# Rodar com: uvicorn main:app --reload

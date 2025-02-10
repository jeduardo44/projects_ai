import openai
from constants import API_KEY

client = openai.OpenAI(api_key=API_KEY)  # Create a client instance

def call_chatgpt(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",  # Use the correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
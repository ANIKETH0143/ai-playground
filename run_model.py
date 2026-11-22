import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

model_path = "./models/smollm2"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)

model.eval()

prompt = "Explain Quantum mechanics in simple words."

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][input_length:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(response)

    
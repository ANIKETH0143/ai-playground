# IMPORTANT: Import spaces first
import spaces

import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ---------------------------------
# 1. Model configuration
# ---------------------------------

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")


# ---------------------------------
# 2. Load model and tokenizer
# ---------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=(
        torch.float16
        if device == "cuda"
        else torch.float32
    )
)

model.to(device)

model.eval()

print("Model loaded successfully!")


# ---------------------------------
# 3. Inference function
# ---------------------------------

@spaces.GPU(duration=60)
def generate_response(prompt: str) -> str:

    if not prompt.strip():
        return "Please enter a prompt."

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
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response


# ---------------------------------
# 4. Gradio interface
# ---------------------------------

demo = gr.Interface(
    fn=generate_response,

    inputs=gr.Textbox(
        label="Ask your AI",
        placeholder="Enter your question...",
        lines=5
    ),

    outputs=gr.Textbox(
        label="AI Response",
        lines=10
    ),

    title="My AI Playground",

    description=(
        "An AI application powered by "
        "SmolLM2 and Hugging Face Transformers."
    )
)


# ---------------------------------
# 5. Launch application
# ---------------------------------

demo.launch()
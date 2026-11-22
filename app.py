import torch
import transformers
import gradio as gr

# print(torch.__version__)
# print(transformers.__version__)


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# -----------------------------
# 1. Load the local model
# -----------------------------

MODEL_PATH = "./models/smollm2"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.to(device)

model.eval()

print("Model loaded successfully!")


# -----------------------------
# 2. Define inference function
# -----------------------------

def generate_response(prompt):

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


# -----------------------------
# 3. Create user interface
# -----------------------------

demo = gr.Interface(
    fn=generate_response,

    inputs=gr.Textbox(
        label="Ask your AI",
        placeholder="Enter your question here...",
        lines=5
    ),

    outputs=gr.Textbox(
        label="AI Response",
        lines=10
    ),

    title="My First AI Playground",

    description=(
        "A locally running LLM powered by "
        "Hugging Face Transformers."
    )
)


# -----------------------------
# 4. Launch application
# -----------------------------

demo.launch()
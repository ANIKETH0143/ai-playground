from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

model.save_pretrained("./models/smollm2")
tokenizer.save_pretrained("./models/smollm2")

print("Model downloaded and saved!")
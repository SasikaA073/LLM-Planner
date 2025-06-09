from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

# Release unused memory
gc.collect()
torch.cuda.empty_cache()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

tokenizer.pad_token = tokenizer.eos_token  # ensure pad_token is set

# Input messages
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Tokenize input
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True).to(device)

# Generate response
with torch.no_grad():
    generated_ids = model.generate(
        inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,         # consider reducing this to avoid OOM
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

# Decode and print
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(output[0])

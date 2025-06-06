from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"
# device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

with torch.no_grad():
    print("This is the begining")
    tokenizer.pad_token = tokenizer.eos_token

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True)

    # Add attention mask explicitly
    input_ids = encodeds.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    model.to(device)

    print("inputs done")

    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,  # use eos as padding
        max_new_tokens=1000,
        do_sample=True
    )

    print("token generation done")
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(decoded[0])

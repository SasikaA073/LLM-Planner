# from https://huggingface.co/mistralai/Pixtral-12B-2409
from vllm import LLM
from vllm.sampling_params import SamplingParams
import os

# Optional: Reduce fragmentation if using NVIDIA GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model configuration
model_name = "mistralai/Pixtral-12B-2409"

# Reduce max_tokens to avoid OOM (adjust as needed)
sampling_params = SamplingParams(
    max_tokens=1024,       # Consider reducing from 8192 to fit GPU memory
    temperature=0.7,
    top_p=0.9,
    stop=["</s>"]
)

# Initialize LLM (vLLM handles GPU inference efficiently)
llm = LLM(
    model=model_name,
    dtype="float16",          # Saves GPU memory
    tokenizer_mode="mistral", # Use Mistral-style chat formatting
    max_model_len=8192        # Optional: Set explicitly if needed
)

# Define prompt and image
prompt = "Describe this image in one sentence."
image_url = "https://picsum.photos/id/237/200/300"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
]

# Generate response
outputs = llm.chat(messages, sampling_params=sampling_params)

# Print result
print(outputs[0].outputs[0].text)

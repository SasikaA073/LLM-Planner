# from https://huggingface.co/microsoft/Florence-2-large-ft
import requests

import torch 
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cuda"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch_dtype=torch.float16

# Clear CUDA cache before loading model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

prompt = "<OD>"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

with torch.no_grad():
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)

import json
import time
import torch

from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM

input_file = "quotes_small.jsonl"

with open(input_file, "r") as f:
    prompts = f.readlines()

    # If the prompts are stored as JSONL, extract the text.
    if input_file.endswith(".jsonl"):
        import json
        prompts = [json.loads(p)["text"] for p in prompts]


model_path = '/scratch/gpfs/DATASETS/bloom_model_1.3/bloom')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder='offload', torch_dtype=torch.bfloat16)

print("Device Map:")
pprint(model.hf_device_map)

def generate(prompt: str):
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs["input_ids"].to(0), min_length=30, max_length=30, do_sample=True)
    print(f"{tokenizer.decode(output[0].tolist()))}: Elapsed Time = {time.time()-t0}")


for prompt in prompts:
    generate(prompt)

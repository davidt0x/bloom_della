import json
import time
import torch

from pprint import pprint
from transformers import AutoTokenizer, pipeline

input_file = "quotes_small.jsonl"

with open(input_file, "r") as f:
    prompts = f.readlines()

    # If the prompts are stored as JSONL, extract the text.
    if input_file.endswith(".jsonl"):
        import json
        prompts = [json.loads(p)["text"] for p in prompts]

model_path = '/scratch/gpfs/DATASETS/bloom_model_1.3/bloom')
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("feature-extraction", model=model_path, device_map="auto", offload_folder='offload', torch_dtype=torch.bfloat16)

print("Device Map:")
pprint(model.hf_device_map)

def embed(prompt: str):
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    ouptut = pipe(inputs["input_ids"].to(0))
    print(f"Elapsed Time = {time.time()-t0}")
    return output


for prompt in prompts:
    embed(prompt)

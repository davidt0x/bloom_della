import json
import time
import torch

from pprint import pprint
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

input_file = "quotes_medium.jsonl"

with open(input_file, "r") as f:
    prompts = f.readlines()

    # If the prompts are stored as JSONL, extract the text.
    if input_file.endswith(".jsonl"):
        import json

        prompts = [json.loads(p)["text"] for p in prompts]

max_gpu_mem = int(60e9)
max_cpu_mem = int(550e9)
max_memory = {
    0: max_gpu_mem,
    1: max_gpu_mem,
    2: max_gpu_mem,
    3: max_gpu_mem,
    "cpu": max_cpu_mem
}

model_path = "/scratch/gpfs/DATASETS/bloom_model_1.3/bloom"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", offload_folder="offload", torch_dtype=torch.bfloat16, max_memory=max_memory,
)

print("Device Map:")
pprint(model.hf_device_map)

for prompt in prompts:
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=1024)
    input = inputs["input_ids"].cuda()
    with torch.no_grad():
        output = model(input)
    print(f"Prompt Size = {input.size()}, Elapsed Time = {time.time()-t0}")

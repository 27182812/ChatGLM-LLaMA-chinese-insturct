import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from src.transformers import AutoTokenizer, AutoConfig, LlamaTokenizer,  LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import json
from dataprocess import format_example


tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained( 
    model, "./qys-alpaca-chinese", torch_dtype=torch.float16
)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""


instructions = json.load(open("data/zh-data01.json"))

answers = []

with torch.no_grad():
    for idx, item in enumerate(instructions[12:18]):
        feature = format_example(item)
        input_text = feature['context']
        print(input_text)
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output.strip())
        print("--------------------------------------------")
    # generation_config = GenerationConfig(
    #         temperature=0.1,
    #         # top_p=0.75,
    #         # top_k=40,
    #         # num_beams=4,
    # )
    # while True:
    #     input_text = input("User:")
    #     inputs = tokenizer(input_text, return_tensors="pt")
    #     input_ids = inputs["input_ids"].cuda()
    #     out = model.generate(
    #         input_ids=input_ids,
    #         generation_config=generation_config,
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #         max_new_tokens=256,
    #     )
    #     s =out.sequences[0]
    #     output = tokenizer.decode(s)
    #     print(output.strip())


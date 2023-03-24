import argparse
import json
from tqdm import tqdm

import datasets
import transformers

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def preprocess(tokenizer, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/zh-data01.json")
    parser.add_argument("--jsonl_path", type=str, default="data/zh-data01.jsonl")
    parser.add_argument("--save_path", type=str, default="data/zh-data01")
    parser.add_argument("--max_seq_length", type=int, default=320)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    with open(args.data_path) as f:
        examples = json.load(f)

    with open(args.jsonl_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            f.write(json.dumps(format_example(example)) + '\n')

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()

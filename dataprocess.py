import argparse
import json
from tqdm import tqdm

import datasets
import transformers


class JsonPreprocessor:
    def __init__(self, data_path, save_path, max_seq_length, skip_overlength):
        self.data_path = data_path
        self.save_path = save_path
        self.max_seq_length = max_seq_length
        self.skip_overlength = skip_overlength
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True
        )

    @staticmethod
    def format_example(example: dict) -> dict:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input']}\n"
        context += "Answer: "
        target = example["output"]
        return {"context": context, "target": target}

    def preprocess(self, example):
        prompt = example["context"]
        target = example["target"]
        prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_seq_length, truncation=True)
        target_ids = self.tokenizer.encode(
            target, max_length=self.max_seq_length, truncation=True, add_special_tokens=False
        )
        input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

    def process(self):
        with open(self.data_path) as f:
            examples = json.load(f)

        features = []
        for example in tqdm(examples, desc="Processing..."):
            formatted_example = self.format_example(example)
            feature = self.preprocess(formatted_example)
            if self.skip_overlength and len(feature["input_ids"]) > self.max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:self.max_seq_length]
            features.append(feature)

        dataset = datasets.Dataset.from_dict({"input_ids": [f["input_ids"] for f in features],"seq_len": [f["seq_len"] for f in features]})
        dataset.save_to_disk(self.save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/zh-data01.json")
    parser.add_argument("--save_path", type=str, default="data/zh-data01")
    parser.add_argument("--max_seq_length", type=int, default=320)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    preprocessor = JsonPreprocessor(
        args.data_path,
        args.save_path,
        args.max_seq_length,
        args.skip_overlength
    )
    preprocessor.process()


if __name__ == "__main__":
    main()

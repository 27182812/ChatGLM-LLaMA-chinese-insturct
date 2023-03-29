import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import json
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from modeling_chatglm import ChatGLMForConditionalGeneration
from dataprocess import format_example

class ChatGLMPredictor:
    def __init__(self, model_path, peft_path, device):
        self.device = device
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.model = ChatGLMForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.load_state_dict(torch.load(peft_path), strict=False)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def predict(self, input_text):
        ids = self.tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).to(self.device)
        out = self.model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = self.tokenizer.decode(out[0])
        return out_text

    def run_prediction(self, instructions):
        answers = []
        with torch.no_grad():
            for idx, item in enumerate(instructions):
                input_text = format_example(item)['context']
                answer = self.predict(input_text)
                item['infer_answer'] = answer
                answers.append({'index': idx, **item})
        return answers


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "THUDM/chatglm-6b"
    peft_path = "output_zh-data01/chatglm-lora.pt"
    instructions_path = "data/zh-data01.json"

    predictor = ChatGLMPredictor(model_path, peft_path, device)
    instructions = json.load(open(instructions_path))

    answers = predictor.run_prediction(instructions[12:18])

    for idx, answer in enumerate(answers):
        print(f"### {idx + 1}. Answer:\n", answer['infer_answer'], '\n\n')
    
    ## interact
    while True:
        input_text = input("User:")
        out_text = predictor.predict(input_text)
        print(out_text)


if __name__ == "__main__":
    main()

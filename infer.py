import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import json
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from modeling_chatglm import ChatGLMForConditionalGeneration
from dataprocess import format_example


torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

peft_path = "output_zh-data01/chatglm-lora.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


instructions = json.load(open("data/zh-data01.json"))

answers = []

with torch.no_grad():
    for idx, item in enumerate(instructions[12:18]):
        feature = format_example(item)
        input_text = feature['context']
        print(input_text)
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).to(device)
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(answer)
        # print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})
    # while True:
    #     input_text = input()
    #     ids = tokenizer.encode(input_text)
    #     input_ids = torch.LongTensor([ids])
    #     out = model.generate(
    #         input_ids=input_ids,
    #         max_length=150,
    #         do_sample=False,
    #         temperature=0
    #     )
    #     out_text = tokenizer.decode(out[0])
    #     print(out_text)
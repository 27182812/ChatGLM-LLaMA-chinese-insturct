# ChatGLM-chinese-insturct

探索中文instruct数据在ChatGLM,LLaMa等LLM上微调表现，结合[PEFT](https://github.com/huggingface/peft)等方法降低资源需求。

基于[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)和[ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)。

##   时间线 / Time line
- [2023-03-28] 完成在中文instruction数据上使用Lora对LLaMa-7B进行微调，发布了[微调后的权重](https://drive.google.com/file/d/1-nqxLz45HkMkhF0NUvkt785MZfPfMld6/view?usp=sharing)。
- [2023-03-24] 完成在中文instruction数据上使用Lora对ChatGLM-6B进行微调，发布了[微调后的权重](https://drive.google.com/file/d/125hjpeS98qum5817XMPp7nY8L19aiOvJ/view?usp=sharing)。

## 样例展示 / Some Examples
### LLaMa-7B
在中文上的效果不如ChatGLM-6B，但考虑其对中文的支持本来就不好，已经不错了（不知道有没有大佬可以尝试增强一下LLaMa的中文能力）

<img width="1440" alt="截屏2023-03-28 10 31 06" src="https://user-images.githubusercontent.com/33630730/228116611-4ca5ffe6-71f5-4401-8e8e-38fc0f5bd575.png">

<img width="600" alt="截屏2023-03-28 10 52 00" src="https://user-images.githubusercontent.com/33630730/228116679-e69d6081-77ff-4a0c-88ed-66fdad9a894a.png">

#### ChatGLM-6B
![截屏2023-03-24 15 54 06](https://user-images.githubusercontent.com/33630730/227459835-a623a86b-5c25-47f9-be06-6e88d4a35e4c.png)

## 环境准备 / Preparing the Enviroment

```bash
conda env create -f env.yml -n bab
conda activate bab
pip install git+https://github.com/huggingface/peft.git
```

## 数据处理 / Processing the Data

Run `bash dataprocess.sh` to process the data.

## 模型微调 / Finetune Your Model

Run `bash finetune.sh` to finetune the model.

##  模型推理 / Inference with Your Model

Run `python infer.py` to do the inference. Show cases in the dataset by default.
 
You can also choose to interact with the model through the annotation section.

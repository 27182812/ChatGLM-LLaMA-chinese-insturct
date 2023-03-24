# ChatGLM-chinese-insturct

探索中文instruct数据在ChatGLM-6B上微调表现，结合[PEFT](https://github.com/huggingface/peft)等方法降低资源需求。

基于[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)和[ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)。


目前已微调好的一版模型权重可自行下载体验（[Google Drive](https://drive.google.com/file/d/125hjpeS98qum5817XMPp7nY8L19aiOvJ/view?usp=sharing))）

## 样例展示 / Some Examples

![截屏2023-03-24 15 54 06](https://user-images.githubusercontent.com/33630730/227459835-a623a86b-5c25-47f9-be06-6e88d4a35e4c.png)

## 环境准备 / Preparing the Enviroment

```bash
conda env create -f env.yml -n bab
conda activate bab
```

## 数据处理 / Processing the Data

Run `bash dataprocess.sh` to process the data.

## 模型微调 / Finetune Your Model

Run `bash finetune.sh` to finetune the model.

##  模型推理 / Inference with Your Model

Run `python infer.py` to do the inference. Show cases in the dataset by default.
 
You can also choose to interact with the model through the annotation section.


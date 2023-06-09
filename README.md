# ChatGLM-LLaMA-chinese-insturct

探索中文instruct数据在ChatGLM, LLaMA等LLM上微调表现，结合[PEFT](https://github.com/huggingface/peft)等方法降低资源需求。

大部分基于[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)、[ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)和[Aplaca-LoRA](https://github.com/tloen/alpaca-lora)，感谢大佬们。

##   时间线 / Time line
- [2023-04-04] 在中文instruction数据上新微调了一版ChatGLM-6B，效果似乎提升了些，发布了[微调后的权重](https://github.com/27182812/ChatGLM-LLaMA-chinese-insturct/tree/main/output_zh-data01)。
- [2023-04-01] 扩充LLaMA的中文词表后，完成在中文instruction数据集[belle](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)上进行微调，发布了[微调后的权重](https://drive.google.com/file/d/12GA0a53DzoKE_dYLSDyBNp-IeCbLgKVn/view?usp=sharing)。
- [2023-03-28] 完成在中文instruction数据上使用Lora对LLaMA-7B进行微调，发布了[微调后的权重](https://drive.google.com/file/d/1-nqxLz45HkMkhF0NUvkt785MZfPfMld6/view?usp=sharing)。
- [2023-03-24] 完成在中文instruction数据上使用Lora对ChatGLM-6B进行微调，发布了[微调后的权重](https://drive.google.com/file/d/125hjpeS98qum5817XMPp7nY8L19aiOvJ/view?usp=sharing)。

## 样例展示 / Some Examples
对于一些生成语句重复现象，可以考虑调整可变参数以及利用规则化的后处理方式去规避。

### ChatGLM-6B
####  ChatGLM-6B-5epoch
感觉这版效果更好，只不过instruction数据后面都会附带一个问题，不过既然格式一样，那就可以想办法规避

<img width="1293" alt="截屏2023-04-04 10 20 31" src="https://user-images.githubusercontent.com/33630730/229670885-a0207b30-f53a-475d-b7ea-bfb6d613beed.png">
<img width="1294" alt="截屏2023-04-04 10 13 38" src="https://user-images.githubusercontent.com/33630730/229670905-b6ca8424-3e54-4446-afad-fe85054b9d2a.png">
<img width="1289" alt="截屏2023-04-04 10 19 55" src="https://user-images.githubusercontent.com/33630730/229670923-8d0284f1-d926-4e82-979f-c9b7f33c2850.png">


####  ChatGLM-6B-3epoch
![截屏2023-03-24 15 54 06](https://user-images.githubusercontent.com/33630730/227459835-a623a86b-5c25-47f9-be06-6e88d4a35e4c.png)

### LLaMa-7B
在中文上的效果不如ChatGLM-6B，但考虑其对中文的支持本来就不好，已经不错了（~~不知道有没有大佬可以尝试增强一下LLaMa的中文能力~~已经有了[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)）

#### LLaMA-7B-belle
注：微调和预测代码和原始一样，但是注意要先根据[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的操作指引合并LoRA权重，生成全量模型权重，这样才是扩充了中文词表后的LLaMA。

![截屏2023-04-01 21 43 34](https://user-images.githubusercontent.com/33630730/229339541-960a3422-a694-4b23-ad0e-16dc7bb491a9.png)
![截屏2023-04-01 21 43 46](https://user-images.githubusercontent.com/33630730/229339544-0084e95e-384c-45fa-9757-1c26c5f4f8e0.png)
![截屏2023-04-01 21 43 54](https://user-images.githubusercontent.com/33630730/229339551-c46db091-91de-41a5-bd52-28b6bc64edf6.png)

#### LLaMA-7B-zh_data01
<img width="1440" alt="截屏2023-03-28 10 31 06" src="https://user-images.githubusercontent.com/33630730/228116611-4ca5ffe6-71f5-4401-8e8e-38fc0f5bd575.png">

<img width="600" alt="截屏2023-03-28 10 52 00" src="https://user-images.githubusercontent.com/33630730/228116679-e69d6081-77ff-4a0c-88ed-66fdad9a894a.png">


## 环境准备 / Preparing the Enviroment

```bash
conda env create -f env.yml -n bab
conda activate bab
pip install git+https://github.com/huggingface/peft.git
```

## 数据处理 / Processing the Data

Run `bash dataprocess.sh` to process the data.

## 模型微调 / Finetune Your Model
### ChatGLM-6B
Run `bash finetune.sh` to finetune the model.

### LLaMA-7B
Run `python test_llama1.py` to finetune the model.

##  模型推理 / Inference with Your Model
You can also choose to interact with the model through the annotation section.

### ChatGLM-6B
Run `python infer.py` to do the inference. Show cases in the dataset by default.

### LLaMA-7B
Run `python generate_llama1.py` to do the inference. Show cases in the dataset by default.

## 友情链接
- [kanchil](https://github.com/vxfla/kanchil): 一个探索小模型的潜力的开源项目



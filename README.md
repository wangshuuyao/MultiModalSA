# 项目README
## 一、项目简介
本项目聚焦于多模态情感分析，旨在解决给定配对的文本和图像，预测其对应的情感标签这一任务，具体为positive（积极）、neutral（中性）、negative（消极）三分类问题。项目使用匿名数据集（实验五数据.zip），该数据集的`data`文件夹包含所有训练文本和图片，以唯一的`guid`命名；`train.txt`文件记录了数据的`guid`及其对应的情感标签；`test_without_label.txt`则存储了数据的`guid`但情感标签为空，需要通过本项目构建的多模态融合模型进行预测。

## 二、代码文件结构
```
lab5
├── data
│   ├── Ablation_Study.py
│   ├── best_model.pth
│   ├── predict.py
│   ├── predictions.txt
│   ├── requirements.txt
│   ├── test_without_label.txt
│   ├── train.py
│   └── train.txt
```

## 三、代码执行流程
1. 确保已安装`requirements.txt`中列出的所有依赖库。
2. 运行`train.py`文件进行模型训练，训练过程会生成模型文件（如`best_model.pth`等）。
3. 训练完成后，运行`prediction.py`文件，使用训练好的模型对`test_without_label.txt`中的数据进行预测，并将预测结果输出到`predictions.txt`文件中。
4. 如需进行消融实验，可运行`Ablation_Study.py`文件。

## 四、依赖库
本项目基于Python 3.9开发，使用以下依赖库实现功能，可通过以下命令进行安装：
```bash
pip install -r requirements.txt
```
具体的依赖版本信息如下：
```
matplotlib==3.10.0
numpy==2.2.2
pandas==2.2.3
Pillow==11.1.0
scikit_learn==1.6.1
seaborn==0.13.2
torch==2.5.1
torchvision==0.20.1
tqdm==4.67.1
transformers==4.47.1
```

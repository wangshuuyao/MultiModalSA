# 项目README

## 一、项目简介
本项目是[描述项目具体内容和目的]，通过代码实现了[具体功能]。

## 二、代码文件结构
```
lab5
├── data
│   ├── Ablation_Study.py
│   ├── best_model.pth
│   ├── multimodal_best_model.pth
│   ├── prediction.py
│   ├── predict.txt
│   ├── requirement.txt
│   ├── test_without_label.txt
│   ├── textual_best_model.pth
│   ├── train.py
│   ├── train.txt
│   └── visual_best_model.pth
```

## 三、代码执行流程
1. 首先确保你已经安装了`requirements.txt`中列出的所有依赖库。
2. 运行`train.py`文件进行模型训练，训练过程中会生成模型文件（如`best_model.pth`等）。
3. 训练完成后，可以运行`prediction.py`文件，使用训练好的模型进行预测。
4. 如需进行消融实验，可运行`Ablation_Study.py`文件。

## 四、依赖库
本项目参考并使用了以下库实现功能，具体的依赖版本信息如下：
```
asttokens==3.0.0
attrs==24.3.0
backcall==0.2.0
beautifulsoup4==4.12.3
bleach==6.2.0
Bottleneck==1.4.2
Brotli==1.0.9
certifi==2024.12.14
charset-normalizer==3.3.2
click==8.1.8
colorama==0.4.6
contourpy==1.2.0
cycler==0.11.0
decorator==5.1.1
defusedxml==0.7.1
docopt==0.6.2
executing==2.1.0
fastjsonschema==2.21.1
filelock==3.13.1
fonttools==4.55.3
fsspec==2024.12.0
gmpy2==2.1.2
huggingface-hub==0.27.0
idna==3.7
imbalanced-learn==0.12.4
imblearn==0.0
importlib_metadata==8.5.0
importlib_resources==6.4.0
ipython==8.12.3
jedi==0.19.2
Jinja2==3.1.4
joblib==1.4.2
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyterlab_pygments==0.3.0
kiwisolver==1.4.4
lightning-utilities==0.11.9
MarkupSafe==2.1.3
matplotlib==3.9.2
matplotlib-inline==0.1.7
mistune==3.1.0
mkl_fft==1.3.11
mkl_random==1.2.8
mkl-service==2.4.0
mpmath==1.3.0
nbclient==0.10.2
nbconvert==7.16.5
nbformat==5.10.4
networkx==3.2.1
nltk==3.9.1
numexpr==2.10.1
numpy==1.26.4
packaging==24.2
pandas==2.2.3
pandocfilters==1.5.1
parso==0.8.4
pickleshare==0.7.5
pillow==11.0.0
pip==24.2
pipreqs==0.5.0
platformdirs==4.3.6
prompt_toolkit==3.0.48
pure_eval==0.2.3
Pygments==2.19.1
pyparsing==3.2.0
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2024.1
pywin32==308
PyYAML==6.0.2
pyzmq==26.2.0
referencing==0.35.1
regex==2024.11.6
requests==2.32.3
rpds-py==0.22.3
safetensors==0.5.0
scikit-learn==1.6.0
scipy==1.13.1
seaborn==0.13.2
sentencepiece==0.2.0
setuptools==75.1.0
six==1.16.0
soupsieve==2.6
stack-data==0.6.3
sympy==1.13.1
threadpoolctl==3.5.0
tinycss2==1.4.0
tokenizers==0.21.0
torch==2.5.1
torchaudio==2.5.1
torchmetrics==1.6.1
torchvision==0.20.1
tornado==6.4.2
tqdm==4.67.1
traitlets==5.14.3
transformers==4.47.1
typing_extensions==4.12.2
tzdata==2023.3
unicodedata2==15.1.0
urllib3==2.2.3
wcwidth==0.2.13
webencodings==0.5.1
wheel==0.44.0
win-inet-pton==1.1.0
yarg==0.1.9
zipp==3.21.0
```

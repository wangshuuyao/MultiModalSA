import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
data_dir = 'data'
train_file = 'train.txt'

# 标签映射
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

from PIL import Image
import torch
from torch.utils.data import Dataset
#定义数据集的读入，包含图像和文本的加载与预处理，并且根据消融要求选取合适的数据集部分（根据description）
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, file_list, tokenizer, transform=None, use_image=True, use_text=True):
        self.data_dir = data_dir
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.transform = transform
        self.use_image = use_image
        self.use_text = use_text

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        guid, label = self.file_list[idx]
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')
        text_path = os.path.join(self.data_dir, f'{guid}.txt')

        image = None
        if self.use_image:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:

                    image = self.transform(image)
                    # print(f"Info: Successed to load image {image_path}")
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}, error: {e}")
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)
        text_inputs = None
        if self.use_text:
            try:
                with open(text_path, 'r', encoding='latin1') as f:
                    text = f.read().strip()
                text_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                text_inputs = {key: val.squeeze(0) for key, val in text_inputs.items()}
            except Exception as e:
                print(f"Warning: Failed to load text {text_path}, error: {e}")
                text_inputs = {'input_ids': torch.zeros(128, dtype=torch.long),
                               'attention_mask': torch.zeros(128, dtype=torch.long)}
        else:
            text_inputs = {'input_ids': torch.zeros(128, dtype=torch.long),
                           'attention_mask': torch.zeros(128, dtype=torch.long)}

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        # # 调试输出
        # print(f"Image: {type(image)}, Text Inputs: {type(text_inputs)}, Label: {label}")
        # print(f"Image Shape: {image.shape if isinstance(image, torch.Tensor) else 'N/A'}")
        # print(f"Text Input IDs Shape: {text_inputs['input_ids'].shape if isinstance(text_inputs, dict) else 'N/A'}")

        return image, text_inputs, label
#多模态模型定义，即训练图像和文本数据，运用resnet+bert
class MultimodalModel(nn.Module):
    def __init__(self, visual_model, textual_model, hidden_size=512):
        super(MultimodalModel, self).__init__()
        self.visual_model = visual_model
        self.textual_model = textual_model

        visual_output_dim = 2048
        textual_output_dim = textual_model.config.hidden_size

        self.fc = nn.Linear(visual_output_dim + textual_output_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)
    #提取图像特征，文本特征，融合图像和文本特征并分类
    def forward(self, image, text_inputs):
        visual_features = self.visual_model(image)

        textual_outputs = self.textual_model(**text_inputs)
        textual_features = textual_outputs.last_hidden_state.mean(dim=1)

        combined_features = torch.cat((visual_features, textual_features), dim=1)

        combined_features = self.fc(combined_features)
        logits = self.classifier(combined_features)

        return logits


# 仅图像输入模型,运用resnet
class VisualOnlyModel(nn.Module):
    def __init__(self, visual_model, hidden_size=512):
        super(VisualOnlyModel, self).__init__()
        self.visual_model = visual_model
        visual_output_dim = 2048
        self.fc = nn.Linear(visual_output_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, image):
        visual_features = self.visual_model(image)
        combined_features = self.fc(visual_features)
        logits = self.classifier(combined_features)
        return logits


# 仅文本输入模型，运用bert
class TextualOnlyModel(nn.Module):
    def __init__(self, textual_model, hidden_size=512):
        super(TextualOnlyModel, self).__init__()
        self.textual_model = textual_model
        textual_output_dim = textual_model.config.hidden_size
        self.fc = nn.Linear(textual_output_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, text_inputs):
        textual_outputs = self.textual_model(**text_inputs)
        textual_features = textual_outputs.last_hidden_state.mean(dim=1)
        combined_features = self.fc(textual_features)
        logits = self.classifier(combined_features)
        return logits


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 读取训练标签
train_data = []
with open(train_file, 'r') as f:
    next(f)  # 跳过标题
    for line in f:
        guid, tag = line.strip().split(',')
        train_data.append((guid, tag))

# 划分训练集与验证集
train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=[tag for _, tag in train_data])

# 模型初始化
visual_model = models.resnet50(weights='IMAGENET1K_V1')  # 使用预训练权重
visual_model.fc = nn.Identity()  # 去除原来的分类层
visual_model = visual_model.to(device)

textual_model = BertModel.from_pretrained('bert-base-uncased')
textual_model = textual_model.to(device)

# 损失函数与优化器
class_weights = torch.tensor([1.0, 3.0, 1.5]).to(device)  # 为不同标签设置权重
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(list(visual_model.parameters()) + list(textual_model.parameters()), lr=1e-5)

# 训练函数
def train(model, train_loader, optimizer, criterion, use_image=True, use_text=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, text_inputs, labels in tqdm(train_loader, desc="Training", unit="batch"):
        if use_image:
            images = images.to(device)
        if use_text:
            text_inputs = {key: value.squeeze().to(device) for key, value in text_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_image and use_text:
            outputs = model(images, text_inputs)
        elif use_image:
            outputs = model(images)
        elif use_text:
            outputs = model(text_inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels

#验证函数，计算 precision, recall, f1-score以及分类报告
def validate(model, val_loader, criterion, use_image=True, use_text=True):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, text_inputs, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            if use_image:
                images = images.to(device)
            if use_text:
                text_inputs = {key: value.squeeze().to(device) for key, value in text_inputs.items()}
            labels = labels.to(device)

            if use_image and use_text:
                outputs = model(images, text_inputs)
            elif use_image:
                outputs = model(images)
            elif use_text:
                outputs = model(text_inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive'])

    return epoch_loss, epoch_acc, precision, recall, f1, report, all_preds, all_labels


# 模型训练和验证循环，三种情况：仅文本，仅图像和融合
for experiment in [ 'multimodal', 'textual', 'visual' ]:
    print(f"Running experiment: {experiment} model")

    if experiment == 'visual':
        model = VisualOnlyModel(visual_model).to(device)
        train_dataset = MultimodalDataset(data_dir, train_data, tokenizer, transform, use_image=True, use_text=False)
        val_dataset = MultimodalDataset(data_dir, val_data, tokenizer, transform, use_image=True, use_text=False)
    elif experiment == 'textual':
        model = TextualOnlyModel(textual_model).to(device)
        train_dataset = MultimodalDataset(data_dir, train_data, tokenizer, transform, use_image=False, use_text=True)
        val_dataset = MultimodalDataset(data_dir, val_data, tokenizer, transform, use_image=False, use_text=True)
    else:
        model = MultimodalModel(visual_model, textual_model).to(device)
        train_dataset = MultimodalDataset(data_dir, train_data, tokenizer, transform, use_image=True, use_text=True)
        val_dataset = MultimodalDataset(data_dir, val_data, tokenizer, transform, use_image=True, use_text=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    best_val_acc = 0.0
    for epoch in range(10):
        train_loss, train_acc, _, _ = train(model, train_loader, optimizer, criterion, use_image=(experiment != 'textual'), use_text=(experiment != 'visual'))
        val_loss, val_acc, val_precision, val_recall, val_f1, val_report, val_preds, val_labels = validate(
            model, val_loader, criterion, use_image=(experiment != 'textual'), use_text=(experiment != 'visual'))

        print(f"Epoch [{epoch+1}/10]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        print("Classification Report:\n", val_report)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{experiment}_best_model.pth')

    print(f"Best Validation Accuracy for {experiment}: {best_val_acc:.4f}")

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix for {experiment} Model")
    plt.show()


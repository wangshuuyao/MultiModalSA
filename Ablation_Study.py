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




# Running experiment: multimodal model
# Training: 100%|██████████| 400/400 [04:11<00:00,  1.59batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.02batch/s]
# Epoch [1/10]
# Train Loss: 0.7998, Train Accuracy: 0.6584
# Val Loss: 0.7472, Val Accuracy: 0.6713
# Val Precision: 0.6787, Val Recall: 0.6713, Val F1: 0.6430
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.74      0.32      0.44       238
#      neutral       0.36      0.38      0.37        84
#     positive       0.71      0.90      0.79       478
#
#     accuracy                           0.67       800
#    macro avg       0.60      0.53      0.53       800
# weighted avg       0.68      0.67      0.64       800
#
# Training: 100%|██████████| 400/400 [04:12<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.06batch/s]
# Epoch [2/10]
# Train Loss: 0.5456, Train Accuracy: 0.7847
# Val Loss: 0.7402, Val Accuracy: 0.7013
# Val Precision: 0.6968, Val Recall: 0.7013, Val F1: 0.6927
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.71      0.52      0.60       238
#      neutral       0.40      0.38      0.39        84
#     positive       0.74      0.85      0.79       478
#
#     accuracy                           0.70       800
#    macro avg       0.62      0.58      0.59       800
# weighted avg       0.70      0.70      0.69       800
#
# Training: 100%|██████████| 400/400 [04:12<00:00,  1.59batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.07batch/s]
# Epoch [3/10]
# Train Loss: 0.3334, Train Accuracy: 0.8859
# Val Loss: 0.7811, Val Accuracy: 0.7037
# Val Precision: 0.6999, Val Recall: 0.7037, Val F1: 0.7011
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.65      0.59      0.62       238
#      neutral       0.40      0.39      0.40        84
#     positive       0.78      0.81      0.79       478
#
#     accuracy                           0.70       800
#    macro avg       0.61      0.60      0.60       800
# weighted avg       0.70      0.70      0.70       800
#
# Training: 100%|██████████| 400/400 [04:12<00:00,  1.59batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.08batch/s]
# Epoch [4/10]
# Train Loss: 0.1882, Train Accuracy: 0.9456
# Val Loss: 0.8896, Val Accuracy: 0.6787
# Val Precision: 0.6906, Val Recall: 0.6787, Val F1: 0.6810
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.65      0.52      0.58       238
#      neutral       0.35      0.50      0.41        84
#     positive       0.77      0.79      0.78       478
#
#     accuracy                           0.68       800
#    macro avg       0.59      0.60      0.59       800
# weighted avg       0.69      0.68      0.68       800
#
# Training: 100%|██████████| 400/400 [04:12<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.01batch/s]
# Epoch [5/10]
# Train Loss: 0.1153, Train Accuracy: 0.9709
# Val Loss: 0.8959, Val Accuracy: 0.7050
# Val Precision: 0.6973, Val Recall: 0.7050, Val F1: 0.6991
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.67      0.58      0.62       238
#      neutral       0.41      0.37      0.39        84
#     positive       0.76      0.83      0.79       478
#
#     accuracy                           0.70       800
#    macro avg       0.61      0.59      0.60       800
# weighted avg       0.70      0.70      0.70       800
#
# Training: 100%|██████████| 400/400 [04:13<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.05batch/s]
# Epoch [6/10]
# Train Loss: 0.0898, Train Accuracy: 0.9794
# Val Loss: 1.0723, Val Accuracy: 0.6887
# Val Precision: 0.6907, Val Recall: 0.6887, Val F1: 0.6851
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.57      0.71      0.63       238
#      neutral       0.44      0.27      0.34        84
#     positive       0.79      0.75      0.77       478
#
#     accuracy                           0.69       800
#    macro avg       0.60      0.58      0.58       800
# weighted avg       0.69      0.69      0.69       800
#
# Training: 100%|██████████| 400/400 [04:13<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.05batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [7/10]
# Train Loss: 0.0706, Train Accuracy: 0.9862
# Val Loss: 0.9593, Val Accuracy: 0.6813
# Val Precision: 0.6869, Val Recall: 0.6813, Val F1: 0.6744
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.72      0.47      0.57       238
#      neutral       0.35      0.40      0.37        84
#     positive       0.73      0.83      0.78       478
#
#     accuracy                           0.68       800
#    macro avg       0.60      0.57      0.57       800
# weighted avg       0.69      0.68      0.67       800
#
# Training: 100%|██████████| 400/400 [04:12<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.08batch/s]
# Epoch [8/10]
# Train Loss: 0.0555, Train Accuracy: 0.9916
# Val Loss: 1.0532, Val Accuracy: 0.6937
# Val Precision: 0.6925, Val Recall: 0.6937, Val F1: 0.6931
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.63      0.65      0.64       238
#      neutral       0.35      0.33      0.34        84
#     positive       0.78      0.78      0.78       478
#
#     accuracy                           0.69       800
#    macro avg       0.59      0.59      0.59       800
# weighted avg       0.69      0.69      0.69       800
#
# Training: 100%|██████████| 400/400 [04:12<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.05batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [9/10]
# Train Loss: 0.0509, Train Accuracy: 0.9900
# Val Loss: 1.0798, Val Accuracy: 0.6887
# Val Precision: 0.6912, Val Recall: 0.6887, Val F1: 0.6896
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.61      0.66      0.63       238
#      neutral       0.35      0.33      0.34        84
#     positive       0.79      0.77      0.78       478
#
#     accuracy                           0.69       800
#    macro avg       0.58      0.59      0.58       800
# weighted avg       0.69      0.69      0.69       800
#
# Training: 100%|██████████| 400/400 [04:13<00:00,  1.58batch/s]
# Validating: 100%|██████████| 100/100 [00:24<00:00,  4.06batch/s]
# Epoch [10/10]
# Train Loss: 0.0419, Train Accuracy: 0.9934
# Val Loss: 1.0788, Val Accuracy: 0.6913
# Val Precision: 0.6801, Val Recall: 0.6913, Val F1: 0.6839
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.64      0.58      0.61       238
#      neutral       0.37      0.30      0.33        84
#     positive       0.75      0.82      0.78       478
#
#     accuracy                           0.69       800
#    macro avg       0.59      0.56      0.57       800
# weighted avg       0.68      0.69      0.68       800
#
# Best Validation Accuracy for multimodal: 0.7050
# Running experiment: textual model
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.87batch/s]
# Epoch [1/10]
# Train Loss: 0.2193, Train Accuracy: 0.9353
# Val Loss: 1.2362, Val Accuracy: 0.6700
# Val Precision: 0.6710, Val Recall: 0.6700, Val F1: 0.6694
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.63      0.55      0.59       238
#      neutral       0.32      0.36      0.34        84
#     positive       0.75      0.78      0.77       478
#
#     accuracy                           0.67       800
#    macro avg       0.57      0.56      0.57       800
# weighted avg       0.67      0.67      0.67       800
#
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.86batch/s]
# Epoch [2/10]
# Train Loss: 0.1264, Train Accuracy: 0.9663
# Val Loss: 1.4743, Val Accuracy: 0.6725
# Val Precision: 0.6710, Val Recall: 0.6725, Val F1: 0.6698
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.65      0.54      0.59       238
#      neutral       0.31      0.33      0.32        84
#     positive       0.75      0.80      0.77       478
#
#     accuracy                           0.67       800
#    macro avg       0.57      0.56      0.56       800
# weighted avg       0.67      0.67      0.67       800
#
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.83batch/s]
# Epoch [3/10]
# Train Loss: 0.1080, Train Accuracy: 0.9697
# Val Loss: 1.5361, Val Accuracy: 0.6925
# Val Precision: 0.6772, Val Recall: 0.6925, Val F1: 0.6728
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.61      0.59      0.60       238
#      neutral       0.54      0.17      0.25        84
#     positive       0.74      0.83      0.78       478
#
#     accuracy                           0.69       800
#    macro avg       0.63      0.53      0.55       800
# weighted avg       0.68      0.69      0.67       800
#
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.87batch/s]
# Epoch [4/10]
# Train Loss: 0.0769, Train Accuracy: 0.9788
# Val Loss: 1.4796, Val Accuracy: 0.7013
# Val Precision: 0.6901, Val Recall: 0.7013, Val F1: 0.6858
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.70      0.50      0.59       238
#      neutral       0.45      0.31      0.37        84
#     positive       0.73      0.87      0.79       478
#
#     accuracy                           0.70       800
#    macro avg       0.63      0.56      0.58       800
# weighted avg       0.69      0.70      0.69       800
#
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.85batch/s]
# Epoch [5/10]
# Train Loss: 0.0626, Train Accuracy: 0.9825
# Val Loss: 1.5355, Val Accuracy: 0.6963
# Val Precision: 0.6868, Val Recall: 0.6963, Val F1: 0.6898
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.67      0.59      0.63       238
#      neutral       0.35      0.30      0.32        84
#     positive       0.75      0.82      0.79       478
#
#     accuracy                           0.70       800
#    macro avg       0.59      0.57      0.58       800
# weighted avg       0.69      0.70      0.69       800
#
# Training: 100%|██████████| 400/400 [02:44<00:00,  2.43batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.85batch/s]
# Epoch [6/10]
# Train Loss: 0.0564, Train Accuracy: 0.9828
# Val Loss: 1.5889, Val Accuracy: 0.6625
# Val Precision: 0.6735, Val Recall: 0.6625, Val F1: 0.6625
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.68      0.50      0.57       238
#      neutral       0.29      0.38      0.33        84
#     positive       0.74      0.79      0.77       478
#
#     accuracy                           0.66       800
#    macro avg       0.57      0.56      0.56       800
# weighted avg       0.67      0.66      0.66       800
#
# Training: 100%|██████████| 400/400 [02:44<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.86batch/s]
# Epoch [7/10]
# Train Loss: 0.0655, Train Accuracy: 0.9781
# Val Loss: 1.7018, Val Accuracy: 0.6887
# Val Precision: 0.6842, Val Recall: 0.6887, Val F1: 0.6863
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.63      0.63      0.63       238
#      neutral       0.36      0.32      0.34        84
#     positive       0.77      0.78      0.77       478
#
#     accuracy                           0.69       800
#    macro avg       0.59      0.58      0.58       800
# weighted avg       0.68      0.69      0.69       800
#
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.85batch/s]
# Epoch [8/10]
# Train Loss: 0.0534, Train Accuracy: 0.9809
# Val Loss: 1.6519, Val Accuracy: 0.6875
# Val Precision: 0.6867, Val Recall: 0.6875, Val F1: 0.6871
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.64      0.63      0.63       238
#      neutral       0.36      0.36      0.36        84
#     positive       0.77      0.77      0.77       478
#
#     accuracy                           0.69       800
#    macro avg       0.59      0.59      0.59       800
# weighted avg       0.69      0.69      0.69       800
#
# Training: 100%|██████████| 400/400 [02:43<00:00,  2.44batch/s]
# Validating: 100%|██████████| 100/100 [00:13<00:00,  7.52batch/s]
# Epoch [9/10]
# Train Loss: 0.0469, Train Accuracy: 0.9825
# Val Loss: 1.6065, Val Accuracy: 0.6887
# Val Precision: 0.6847, Val Recall: 0.6887, Val F1: 0.6847
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.68      0.57      0.62       238
#      neutral       0.33      0.32      0.32        84
#     positive       0.75      0.81      0.78       478
#
#     accuracy                           0.69       800
#    macro avg       0.58      0.57      0.57       800
# weighted avg       0.68      0.69      0.68       800
#
# Training: 100%|██████████| 400/400 [02:44<00:00,  2.43batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.84batch/s]
# Epoch [10/10]
# Train Loss: 0.0465, Train Accuracy: 0.9816
# Val Loss: 1.8411, Val Accuracy: 0.6813
# Val Precision: 0.6723, Val Recall: 0.6813, Val F1: 0.6741
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.66      0.55      0.60       238
#      neutral       0.31      0.27      0.29        84
#     positive       0.74      0.82      0.78       478
#
#     accuracy                           0.68       800
#    macro avg       0.57      0.55      0.56       800
# weighted avg       0.67      0.68      0.67       800
#
# Best Validation Accuracy for textual: 0.7013
# Running experiment: visual model
# Training: 100%|██████████| 400/400 [01:34<00:00,  4.21batch/s]
# Validating: 100%|██████████| 100/100 [00:13<00:00,  7.61batch/s]
# Epoch [1/10]
# Train Loss: 0.5837, Train Accuracy: 0.8069
# Val Loss: 1.2247, Val Accuracy: 0.5325
# Val Precision: 0.6035, Val Recall: 0.5325, Val F1: 0.5530
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.44      0.50      0.47       238
#      neutral       0.25      0.54      0.34        84
#     positive       0.75      0.55      0.63       478
#
#     accuracy                           0.53       800
#    macro avg       0.48      0.53      0.48       800
# weighted avg       0.60      0.53      0.55       800
#
# Training: 100%|██████████| 400/400 [01:37<00:00,  4.12batch/s]
# Validating: 100%|██████████| 100/100 [00:13<00:00,  7.57batch/s]
# Epoch [2/10]
# Train Loss: 0.2843, Train Accuracy: 0.9241
# Val Loss: 1.1809, Val Accuracy: 0.5962
# Val Precision: 0.6140, Val Recall: 0.5962, Val F1: 0.6028
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.49      0.45      0.47       238
#      neutral       0.30      0.46      0.37        84
#     positive       0.73      0.69      0.71       478
#
#     accuracy                           0.60       800
#    macro avg       0.51      0.53      0.51       800
# weighted avg       0.61      0.60      0.60       800
#
# Training: 100%|██████████| 400/400 [01:38<00:00,  4.06batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.74batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [3/10]
# Train Loss: 0.2223, Train Accuracy: 0.9416
# Val Loss: 1.2930, Val Accuracy: 0.5475
# Val Precision: 0.5995, Val Recall: 0.5475, Val F1: 0.5641
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.45      0.48      0.47       238
#      neutral       0.27      0.52      0.36        84
#     positive       0.73      0.58      0.65       478
#
#     accuracy                           0.55       800
#    macro avg       0.48      0.53      0.49       800
# weighted avg       0.60      0.55      0.56       800
#
# Training: 100%|██████████| 400/400 [01:36<00:00,  4.13batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.81batch/s]
# Epoch [4/10]
# Train Loss: 0.1687, Train Accuracy: 0.9591
# Val Loss: 1.2025, Val Accuracy: 0.5813
# Val Precision: 0.6032, Val Recall: 0.5813, Val F1: 0.5905
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.48      0.47      0.48       238
#      neutral       0.25      0.36      0.29        84
#     positive       0.73      0.67      0.70       478
#
#     accuracy                           0.58       800
#    macro avg       0.48      0.50      0.49       800
# weighted avg       0.60      0.58      0.59       800
#
# Training: 100%|██████████| 400/400 [01:34<00:00,  4.21batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  8.14batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [5/10]
# Train Loss: 0.1500, Train Accuracy: 0.9616
# Val Loss: 1.2331, Val Accuracy: 0.5675
# Val Precision: 0.5821, Val Recall: 0.5675, Val F1: 0.5736
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.44      0.42      0.43       238
#      neutral       0.28      0.38      0.32        84
#     positive       0.71      0.67      0.69       478
#
#     accuracy                           0.57       800
#    macro avg       0.47      0.49      0.48       800
# weighted avg       0.58      0.57      0.57       800
#
# Training: 100%|██████████| 400/400 [01:36<00:00,  4.15batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.85batch/s]
# Epoch [6/10]
# Train Loss: 0.1412, Train Accuracy: 0.9659
# Val Loss: 1.1133, Val Accuracy: 0.6000
# Val Precision: 0.5853, Val Recall: 0.6000, Val F1: 0.5846
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.49      0.32      0.38       238
#      neutral       0.30      0.35      0.32        84
#     positive       0.68      0.79      0.73       478
#
#     accuracy                           0.60       800
#    macro avg       0.49      0.48      0.48       800
# weighted avg       0.59      0.60      0.58       800
#
# Training: 100%|██████████| 400/400 [01:36<00:00,  4.15batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.80batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [7/10]
# Train Loss: 0.1184, Train Accuracy: 0.9712
# Val Loss: 1.2612, Val Accuracy: 0.5700
# Val Precision: 0.5802, Val Recall: 0.5700, Val F1: 0.5745
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.44      0.43      0.44       238
#      neutral       0.28      0.36      0.32        84
#     positive       0.70      0.68      0.69       478
#
#     accuracy                           0.57       800
#    macro avg       0.48      0.49      0.48       800
# weighted avg       0.58      0.57      0.57       800
#
# Training: 100%|██████████| 400/400 [01:37<00:00,  4.11batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.77batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [8/10]
# Train Loss: 0.1188, Train Accuracy: 0.9728
# Val Loss: 1.1603, Val Accuracy: 0.6000
# Val Precision: 0.5938, Val Recall: 0.6000, Val F1: 0.5954
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.48      0.40      0.44       238
#      neutral       0.33      0.37      0.35        84
#     positive       0.70      0.74      0.72       478
#
#     accuracy                           0.60       800
#    macro avg       0.50      0.50      0.50       800
# weighted avg       0.59      0.60      0.60       800
#
# Training: 100%|██████████| 400/400 [01:36<00:00,  4.13batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  7.87batch/s]
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch [9/10]
# Train Loss: 0.1197, Train Accuracy: 0.9688
# Val Loss: 1.1203, Val Accuracy: 0.5988
# Val Precision: 0.6039, Val Recall: 0.5988, Val F1: 0.5973
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.50      0.38      0.43       238
#      neutral       0.29      0.43      0.35        84
#     positive       0.71      0.74      0.72       478
#
#     accuracy                           0.60       800
#    macro avg       0.50      0.52      0.50       800
# weighted avg       0.60      0.60      0.60       800
#
# Training: 100%|██████████| 400/400 [01:34<00:00,  4.24batch/s]
# Validating: 100%|██████████| 100/100 [00:12<00:00,  8.29batch/s]
# Epoch [10/10]
# Train Loss: 0.0937, Train Accuracy: 0.9806
# Val Loss: 1.3309, Val Accuracy: 0.5850
# Val Precision: 0.6068, Val Recall: 0.5850, Val F1: 0.5938
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.47      0.50      0.48       238
#      neutral       0.30      0.40      0.34        84
#     positive       0.73      0.66      0.69       478
#
#     accuracy                           0.58       800
#    macro avg       0.50      0.52      0.51       800
# weighted avg       0.61      0.58      0.59       800
#
# Best Validation Accuracy for visual: 0.6000



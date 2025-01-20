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

# 数据集类，进行图像和文本的加载与预处理
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, file_list, tokenizer, transform=None):
        self.data_dir = data_dir
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        guid, label = self.file_list[idx]
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')
        text_path = os.path.join(self.data_dir, f'{guid}.txt')

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        with open(text_path, 'r', encoding='latin1') as f:
            text = f.read().strip()
        text_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        label = label_map[label]

        return image, text_inputs, label

# 图像和文本模型融合
class MultimodalModel(nn.Module):
    def __init__(self, visual_model, textual_model, hidden_size=512):
        super(MultimodalModel, self).__init__()
        self.visual_model = visual_model
        self.textual_model = textual_model

        visual_output_dim = 2048
        textual_output_dim = textual_model.config.hidden_size

        self.fc = nn.Linear(visual_output_dim + textual_output_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)
    #提取图像特征和文本特征，并且融合图像与文本特征，进行分类
    def forward(self, image, text_inputs):
        visual_features = self.visual_model(image)

        textual_outputs = self.textual_model(**text_inputs)
        textual_features = textual_outputs.last_hidden_state.mean(dim=1)

        combined_features = torch.cat((visual_features, textual_features), dim=1)

        combined_features = self.fc(combined_features)
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

train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=[tag for _, tag in train_data])

train_dataset = MultimodalDataset(data_dir, train_data, tokenizer, transform)
val_dataset = MultimodalDataset(data_dir, val_data, tokenizer, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 模型初始化
visual_model = models.resnet50(weights='IMAGENET1K_V1')
visual_model.fc = nn.Identity()
visual_model = visual_model.to(device)

textual_model = BertModel.from_pretrained('bert-base-uncased')
textual_model = textual_model.to(device)

model = MultimodalModel(visual_model, textual_model).to(device)

# 损失函数与优化器，为不同标签设置权重
class_weights = torch.tensor([1.0, 3.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, text_inputs, labels in tqdm(train_loader, desc="Training", unit="batch"):
        images = images.to(device)
        text_inputs = {key: value.squeeze().to(device) for key, value in text_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, text_inputs)
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

# 验证函数,计算precision, recall, f1-score,计算分类报告
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, text_inputs, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images = images.to(device)
            text_inputs = {key: value.squeeze().to(device) for key, value in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(images, text_inputs)
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

best_acc = 0.0
early_stopping_counter = 0
early_stopping_patience = 3

#运行十个epoch, 保存最优模型,加载并评估最优模型
for epoch in range(10):
    print(f"Epoch {epoch + 1}/10:")

    train_loss, train_acc, _, _ = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc, precision, recall, f1, report, all_preds, all_labels = validate(model, val_loader, criterion)

    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("  Classification Report:")
    print(report)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")
        early_stopping_counter = 0  # 重置早停计数器
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered!")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
# Epoch 1/10:
# Training: 100%|██████████| 400/400 [03:51<00:00,  1.73batch/s]
# Validating: 100%|██████████| 100/100 [00:25<00:00,  3.89batch/s]
#   Train Loss: 0.7875, Train Accuracy: 0.6587
#   Validation Loss: 0.6598, Validation Accuracy: 0.7137
#   Precision: 0.7401, Recall: 0.7137, F1-Score: 0.7132
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.79      0.49      0.61       238
#      neutral       0.37      0.58      0.45        84
#     positive       0.78      0.85      0.81       478
#
#     accuracy                           0.71       800
#    macro avg       0.65      0.64      0.62       800
# weighted avg       0.74      0.71      0.71       800
#
# Model saved!
# C:\Users\Sherria\Desktop\2024秋作业\人工智能\lab5\8.py:224: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('best_model.pth'))
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch 2/10:
# Training: 100%|██████████| 400/400 [04:11<00:00,  1.59batch/s]
# Validating: 100%|██████████| 100/100 [00:26<00:00,  3.73batch/s]
#   Train Loss: 0.5299, Train Accuracy: 0.7869
#   Validation Loss: 0.6561, Validation Accuracy: 0.7488
#   Precision: 0.7464, Recall: 0.7488, F1-Score: 0.7368
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.79      0.56      0.66       238
#      neutral       0.56      0.42      0.48        84
#     positive       0.76      0.90      0.82       478
#
#     accuracy                           0.75       800
#    macro avg       0.70      0.63      0.65       800
# weighted avg       0.75      0.75      0.74       800
#
# Model saved!
# C:\Users\Sherria\Desktop\2024秋作业\人工智能\lab5\8.py:224: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('best_model.pth'))
# Epoch 3/10:
# Training: 100%|██████████| 400/400 [04:20<00:00,  1.53batch/s]
# Validating: 100%|██████████| 100/100 [00:26<00:00,  3.71batch/s]
#   Train Loss: 0.2648, Train Accuracy: 0.9038
#   Validation Loss: 0.8060, Validation Accuracy: 0.7400
#   Precision: 0.7367, Recall: 0.7400, F1-Score: 0.7353
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.74      0.60      0.66       238
#      neutral       0.48      0.46      0.47        84
#     positive       0.78      0.86      0.82       478
#
#     accuracy                           0.74       800
#    macro avg       0.67      0.64      0.65       800
# weighted avg       0.74      0.74      0.74       800
#
# C:\Users\Sherria\Desktop\2024秋作业\人工智能\lab5\8.py:224: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('best_model.pth'))
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch 4/10:
# Training: 100%|██████████| 400/400 [04:25<00:00,  1.51batch/s]
# Validating: 100%|██████████| 100/100 [00:26<00:00,  3.72batch/s]
#   Train Loss: 0.2622, Train Accuracy: 0.8959
#   Validation Loss: 0.7839, Validation Accuracy: 0.7550
#   Precision: 0.7628, Recall: 0.7550, F1-Score: 0.7582
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.71      0.74      0.72       238
#      neutral       0.47      0.55      0.51        84
#     positive       0.84      0.80      0.82       478
#
#     accuracy                           0.76       800
#    macro avg       0.67      0.70      0.68       800
# weighted avg       0.76      0.76      0.76       800
#
# Model saved!
# C:\Users\Sherria\Desktop\2024秋作业\人工智能\lab5\8.py:224: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('best_model.pth'))
# Training:   0%|          | 0/400 [00:00<?, ?batch/s]Epoch 5/10:
# Training: 100%|██████████| 400/400 [04:30<00:00,  1.48batch/s]
# Validating: 100%|██████████| 100/100 [00:27<00:00,  3.61batch/s]
#   Train Loss: 0.1188, Train Accuracy: 0.9600
#   Validation Loss: 0.9628, Validation Accuracy: 0.7362
#   Precision: 0.7460, Recall: 0.7362, F1-Score: 0.7348
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.78      0.57      0.66       238
#      neutral       0.41      0.52      0.46        84
#     positive       0.79      0.86      0.82       478
#
#     accuracy                           0.74       800
#    macro avg       0.66      0.65      0.65       800
# weighted avg       0.75      0.74      0.73       800
#
# C:\Users\Sherria\Desktop\2024秋作业\人工智能\lab5\8.py:224: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('best_model.pth'))
# Epoch 6/10:
# Training: 100%|██████████| 400/400 [04:31<00:00,  1.47batch/s]
# Validating: 100%|██████████| 100/100 [00:27<00:00,  3.63batch/s]
# C:\Users\Sherria\Desktop\2024秋作业\人工智能\lab5\8.py:224: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('best_model.pth'))
#   Train Loss: 0.1303, Train Accuracy: 0.9569
#   Validation Loss: 0.9751, Validation Accuracy: 0.7425
#   Precision: 0.7404, Recall: 0.7425, F1-Score: 0.7413
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.70      0.68      0.69       238
#      neutral       0.45      0.44      0.45        84
#     positive       0.81      0.83      0.82       478
#
#     accuracy                           0.74       800
#    macro avg       0.65      0.65      0.65       800
# weighted avg       0.74      0.74      0.74       800
#
# Epoch 7/10:
# Training: 100%|██████████| 400/400 [04:32<00:00,  1.47batch/s]
# Validating: 100%|██████████| 100/100 [00:28<00:00,  3.57batch/s]
#   Train Loss: 0.1194, Train Accuracy: 0.9616
#   Validation Loss: 0.9558, Validation Accuracy: 0.7312
#   Precision: 0.7377, Recall: 0.7312, F1-Score: 0.7285
#   Classification Report:
#               precision    recall  f1-score   support
#
#     negative       0.75      0.55      0.64       238
#      neutral       0.43      0.54      0.48        84
#     positive       0.78      0.85      0.82       478
#
#     accuracy                           0.73       800
#    macro avg       0.66      0.65      0.64       800
# weighted avg       0.74      0.73      0.73       800
#
# Early stopping triggered!

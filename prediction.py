import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader
import os


# 定义多模态融合模型（与训练时相同）
class MultiModalModel(torch.nn.Module):
    def __init__(self, image_model, text_model, hidden_dim=512, num_classes=3):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        visual_output_dim = 2048
        textual_output_dim = text_model.config.hidden_size
        self.fc1 = torch.nn.Linear(visual_output_dim + textual_output_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, image, text_inputs):
        image_features = self.image_model(image)
        textual_outputs = self.text_model(**text_inputs)
        text_features = textual_outputs.last_hidden_state.mean(dim=1)  # 使用 BERT 的[CLS]标记作为文本特征
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x


# 数据预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 载入 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 加载训练好的模型
def load_model():
    resnet = models.resnet50(weights='IMAGENET1K_V1')
    resnet.fc = torch.nn.Identity()
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = MultiModalModel(image_model=resnet, text_model=bert)

    state_dict = torch.load('best_model.pth', weights_only=True)
    new_state_dict = {}
    for key in state_dict.keys():
        if 'visual_model' in key:
            new_state_dict[key.replace('visual_model', 'image_model')] = state_dict[key]
        elif 'textual_model' in key:
            new_state_dict[key.replace('textual_model', 'text_model')] = state_dict[key]
        elif 'fc' in key or 'classifier' in key:
            continue
        else:
            new_state_dict[key] = state_dict[key]
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


# 读取 test_without_label.txt 文件并进行预测
def predict_and_save(model, test_file, image_dir, output_file):
    test_df = pd.read_csv(test_file, header=0)
    predictions = []

    for idx, row in test_df.iterrows():
        guid = row['guid']

        guid = str(int(guid))

        img_path = os.path.join(image_dir, f"{guid}.jpg")
        txt_path = os.path.join(image_dir, f"{guid}.txt")

        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found.")
            predictions.append('unknown')
            continue

        if not os.path.exists(txt_path):
            print(f"Warning: Text file {txt_path} not found.")
            predictions.append('unknown')
            continue

        image = Image.open(img_path).convert("RGB")
        image = image_transform(image).unsqueeze(0)

        with open(txt_path, 'r', encoding='latin1') as f:
            text = f.read()
        text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

        with torch.no_grad():
            output = model(image, text_inputs)
            _, predicted = torch.max(output, 1)

        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_label = label_map[predicted.item()]
        predictions.append(predicted_label)

    test_df['tag'] = predictions
    test_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# 主要执行逻辑
if __name__ == "__main__":
    model = load_model()  # 加载模型
    test_file = 'test_without_label.txt'  # 测试集文件路径
    image_dir = 'data'  # 图像和文本文件所在目录
    output_file = 'predictions.txt'  # 预测结果保存路径

    predict_and_save(model, test_file, image_dir, output_file)
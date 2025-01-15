# cluster微调

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, AdamW
import json


# 模拟数据集
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_data_from_jsonl(path):
    with open(path, "r") as f:
        data = f.readlines()
    texts = []
    labels = []
    for line in data:
        line = json.loads(line)
        texts.append(line["prompt"])
        labels.append(line["type"])
    return texts, labels

questions,types = load_data_from_jsonl("/ceph/home/yangshu/lema/final_project/test_data/math_test.jsonl")

# 初始化数据集和数据加载器
texts = questions
labels = []
# label0-7对应type algebra等等
for i in range(len(types)):
    if types[i] == "Algebra":
        labels.append(0)
    elif types[i] == "Counting & Probability": 
        labels.append(1)
    elif types[i] == "Geometry":
        labels.append(2)
    elif types[i] == "Number Theory":
        labels.append(3)
    elif types[i] == "Intermediate Algebra":    
        labels.append(4)
    elif types[i] == "Prealgebra":
        labels.append(5)
    elif types[i] == "Precalculus":
        labels.append(6)
    



tokenizer = BertTokenizer.from_pretrained("/ceph/home/yangshu/lema/final_project/mathbert")
dataset = CustomDataset(texts, labels, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 初始化模型
class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("/ceph/home/yangshu/lema/final_project/mathbert")
        self.classifier = nn.Linear(self.bert.config.hidden_size, 7)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出作为句子级别 embedding
        logits = self.classifier(embeddings)  # 分类任务输出
        return logits, embeddings


model = CustomBERTModel()
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# 类别中心初始化
num_classes = 7
hidden_size = model.bert.config.hidden_size
class_centers = torch.zeros(num_classes, hidden_size, requires_grad=False).to("cuda")

# 训练
model = model.to("cuda")
for epoch in range(3):  # 假设训练 3 个 epoch
    for batch in dataloader:
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        logits, embeddings = model(input_ids, attention_mask)

        # 计算任务损失
        task_loss = criterion(logits, labels)

        # 更新类别中心
        with torch.no_grad():
            for cls in range(num_classes):
                mask = labels == cls
                if mask.any():
                    class_embeddings = embeddings[mask]
                    class_centers[cls] = class_embeddings.mean(dim=0)

        # 计算类别中心损失
        distances = torch.norm(embeddings - class_centers[labels], dim=1)
        cluster_loss = distances.mean()

        # 总损失
        loss = task_loss + 0.1 * cluster_loss

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
    
    # 保存模型
    model_save_path = "/ceph/home/yangshu/lema/finetune/checkpoint_0"

    torch.save(model.state_dict(), f"{model_save_path}/model_state_{epoch}.pt")



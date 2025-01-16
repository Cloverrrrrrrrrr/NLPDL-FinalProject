# cluster Finetune

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, AdamW
import json
import argparse
import os

# 定义数据集类
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

# 从JSONL文件加载数据
def load_data_from_jsonl(path):
    with open(path, "r") as f:
        data = f.readlines()
    texts = []
    labels = []
    for line in data:
        line = json.loads(line)
        texts.append(line["problem"])
        labels.append(line["type"])
    return texts, labels

# 定义模型类
class CustomBERTModel(nn.Module):
    def __init__(self, model_path, num_classes):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出作为句子级别 embedding
        logits = self.classifier(embeddings)  # 分类任务输出
        return logits, embeddings

# 主逻辑
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Cluster Fine-tuning with BERT")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained BERT model")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the model checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # 加载数据
    questions, types = load_data_from_jsonl(args.data_path)
    type_mapping = {
        "Algebra": 0,
        "Counting & Probability": 1,
        "Geometry": 2,
        "Number Theory": 3,
        "Intermediate Algebra": 4,
        "Prealgebra": 5,
        "Precalculus": 6
    }
    labels = [type_mapping.get(t, -1) for t in types]  # 默认-1表示未知类别

    # 初始化数据集和数据加载器
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    dataset = CustomDataset(questions, labels, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    num_classes = len(type_mapping)
    model = CustomBERTModel(args.model_path, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 类别中心初始化
    hidden_size = model.bert.config.hidden_size
    class_centers = torch.zeros(num_classes, hidden_size, requires_grad=False).to("cuda")

    # 训练
    model = model.to("cuda")
    os.makedirs(args.save_path, exist_ok=True)  # 确保保存路径存在
    for epoch in range(args.epochs):
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
        torch.save(model.state_dict(), os.path.join(args.save_path, f"model_state_{epoch}.pt"))




# triple loss 微调

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers.optimization import AdamW
import json
import torch.nn as nn
import os
import torch.nn.functional as F

class QueryDocumentDataset(Dataset):
    def __init__(self, queries, positive_docs, negative_docs, tokenizer, max_len=512):
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        positive_doc = self.positive_docs[idx]
        negative_doc = self.negative_docs[idx]
        
        # 编码查询和文档
        query_encoding = self.tokenizer.encode_plus(query, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        positive_encoding = self.tokenizer.encode_plus(positive_doc, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        negative_encoding = self.tokenizer.encode_plus(negative_doc, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
        }

def load_data_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problems.append(data)
    return problems

queries_raw = load_data_from_jsonl("/ceph/home/yangshu/lema/final_project/test_data/math_test.jsonl")
raw_docs = load_data_from_jsonl("/ceph/home/yangshu/lema/final_project/train_data/math/dataset.jsonl")

# 在raw_docs中提取和queries type相同的文档作为正例
docs_positive = []
cnt = 0
for query in queries_raw:
    query_type = query['type']
    while raw_docs[cnt % 7500]['type'] != query_type:
        cnt += 1
    docs_positive.append(raw_docs[cnt % 7500])
    cnt += 1

# 在raw_docs中提取和queries type不同的文档作为负例
docs_negative = []
cnt = 0
for query in queries_raw:
    query_type = query['type']
    while raw_docs[cnt % 7500]['type'] == query_type:
        cnt += 1
    docs_negative.append(raw_docs[cnt])
    cnt += 1

queries = [query['prompt'] for query in queries_raw]
docs_positive = [doc['problem'] for doc in docs_positive]
docs_negative = [doc['problem'] for doc in docs_negative]


labels_positive = [1] * len(queries)
labels_negative = [0] * len(queries)

model_path = "/ceph/home/yangshu/lema/final_project/mathbert"


# 初始化Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 创建数据集
dataset = QueryDocumentDataset(queries, docs_positive, docs_negative, tokenizer)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class BERTEmbedder(nn.Module):
    def __init__(self, model_name=model_path):
        super(BERTEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooler = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 取[CLS] token的输出
        return self.pooler(embeddings)  # 使用一个pooling层进行投影



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # 计算欧氏距离
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # 计算三元组损失
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()

model = BERTEmbedder()
triple_loss = TripletLoss(margin=1.0)
optimizer = AdamW(model.parameters(), lr=1e-5)

# 训练模型
model.train()
for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()
        
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']
        positive_input_ids = batch['positive_input_ids']
        positive_attention_mask = batch['positive_attention_mask']
        negative_input_ids = batch['negative_input_ids']
        negative_attention_mask = batch['negative_attention_mask']
        
        query_embedding = model(query_input_ids, query_attention_mask)
        positive_embedding = model(positive_input_ids, positive_attention_mask)
        negative_embedding = model(negative_input_ids, negative_attention_mask)
        
        loss = triple_loss(query_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
    
    # 保存模型
    # 文件夹
    os.makedirs(f"/ceph/home/yangshu/lema/finetune/checkpoint_{epoch}", exist_ok=True)
    # 保存模型
    model.save_pretrained(f"/ceph/home/yangshu/lema/finetune/checkpoint_{epoch}")

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Triple Loss Finetune

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers.optimization import AdamW
import json
import torch.nn as nn
import os
import torch.nn.functional as F
import argparse

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

class BERTEmbedder(nn.Module):
    def __init__(self, model_name):
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

def load_data_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problems.append(data)
    return problems

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_data', type=str, required=True, help="Path to query data JSONL file")
    parser.add_argument('--doc_data', type=str, required=True, help="Path to document data JSONL file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to pre-trained model directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")

    args = parser.parse_args()

    # 加载数据
    queries_raw = load_data_from_jsonl(args.query_data)
    raw_docs = load_data_from_jsonl(args.doc_data)

    # 提取正负例
    docs_positive, docs_negative = [], []
    cnt = 0
    for query in queries_raw:
        query_type = query['type']
        while raw_docs[cnt % len(raw_docs)]['type'] != query_type:
            cnt += 1
        docs_positive.append(raw_docs[cnt % len(raw_docs)])
        cnt += 1

    cnt = 0
    for query in queries_raw:
        query_type = query['type']
        while raw_docs[cnt % len(raw_docs)]['type'] == query_type:
            cnt += 1
        docs_negative.append(raw_docs[cnt % len(raw_docs)])
        cnt += 1

    queries = [query['problem'] for query in queries_raw]
    docs_positive = [doc['problem'] for doc in docs_positive]
    docs_negative = [doc['problem'] for doc in docs_negative]

    # 初始化Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # 创建数据集和DataLoader
    dataset = QueryDocumentDataset(queries, docs_positive, docs_negative, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型和优化器
    model = BERTEmbedder(args.model_path)
    triple_loss = TripletLoss(margin=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 训练模型
    model.train()
    for epoch in range(args.epochs):
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
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

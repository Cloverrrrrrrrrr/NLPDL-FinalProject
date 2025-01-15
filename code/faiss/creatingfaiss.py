import json
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig, AutoConfig
import torch
import faiss

import argparse

def load_data_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problems.append(data)
    return problems

def load_model(model_path, state_dict_path):
    checkpoint = torch.load(state_dict_path)
    config = BertConfig.from_pretrained(model_path)
    model = BertModel(config)
    model.load_state_dict(checkpoint, strict=False)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--state_dict_path", type=str, help="Path to the state dict")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--document_path", type=str, help="Path to the document file")
    
    args = parser.parse_args()

    model_path = args.model_path
    state_dict_path = args.state_dict_path
    output_file = args.output_file
    document_path = args.document_path

    # 加载文档
    load_data_from_jsonl(document_path)
    # 创建问题与答案的文本列表
    questions = [entry['problem'] for entry in corrections]

    # 加载模型
    model, tokenizer = load_model(model_path, state_dict_path)

    # 获取问题和答案的嵌入

    question_embeddings = []
    batch_size = 32
    for i in range(0, len(questions), batch_size):
        embeddings = get_embeddings(questions[i:i + batch_size], model, tokenizer)
        question_embeddings.extend(embeddings)

    # 嵌入共有7500条，每条嵌入维度为768
    question_embeddings = torch.cat(question_embeddings, dim=0).numpy()
    question_embeddings = question_embeddings.reshape(7500, 768)

    # 查看向量形状
    print(np.shape(question_embeddings))  # (n, embedding_dim)

    # 向量的维度 (根据 SentenceTransformer 的嵌入大小)
    embedding_dim = question_embeddings.shape[1]

    # 创建 FAISS 索引
    index = faiss.IndexFlatL2(embedding_dim)  # 使用 L2 距离

    # 将问题和答案的向量添加到索引中
    index.add(np.array(question_embeddings, dtype=np.float32))


    # 查看索引的大小
    print("Number of vectors in the index:", index.ntotal)

    # 保存 FAISS 索引
    faiss.write_index(index, output_file)

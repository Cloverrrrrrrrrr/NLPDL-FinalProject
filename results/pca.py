import faiss
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


# 加载FAISS索引和文档
def load_faiss_index_and_documents(index_path, documents_path):
    # 读取FAISS索引
    index = faiss.read_index(index_path)
    
    # 读取文档
    documents = []
    with open(documents_path, 'r') as file:
        cnt = 0
        for line in file:
            data = json.loads(line)
            # 给每个文档添加id
            data['id'] = cnt
            cnt += 1
            documents.append(data)
    
    return index, documents

# 提取文档特征向量
def extract_feature_vectors(index, num_docs):
    feature_vectors = []
    for i in range(num_docs):
        vector = index.reconstruct(i)  # 从FAISS索引中提取特征向量
        feature_vectors.append(vector)
    return feature_vectors

# 根据type分类文档
def classify_documents_by_type(documents):
    classified_docs = {}
    for doc in documents:
        doc_type = doc['type']
        if doc_type not in classified_docs:
            classified_docs[doc_type] = []
        classified_docs[doc_type].append(doc)
    return classified_docs

# 按type分类特征向量
def classify_feature_vectors_by_type(feature_vectors, classified_docs):
    classified_vectors = {}
    for doc_type, docs in classified_docs.items():
        classified_vectors[doc_type] = []
        for doc in docs:
            doc_id = doc['id']
            vector = feature_vectors[doc_id]
            classified_vectors[doc_type].append(vector)
    return classified_vectors

# 示例主函数
def main(index_path, documents_path, output_file):
    # 加载FAISS索引和文档
    index, documents = load_faiss_index_and_documents(index_path, documents_path)
    
    # 提取特征向量
    feature_vectors = extract_feature_vectors(index, len(documents))
    
    # 根据type分类文档
    classified_docs = classify_documents_by_type(documents)
    
    # 按type分类特征向量
    classified_vectors = classify_feature_vectors_by_type(feature_vectors, classified_docs)
    
    # 按照type进行pca
    for doc_type, vectors in classified_vectors.items():
        # 进行PCA
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)
        
        # 绘制散点图
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], label=doc_type)

        
    
    plt.legend()
    plt.savefig(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, required=True, help='Path to the FAISS index file')
    parser.add_argument('--documents_path', type=str, required=True, help='Path to the document file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()
    
    index_path = args.index_path
    documents_path = args.documents_path
    output_file = args.output_file

    main(index_path, documents_path, output_file)


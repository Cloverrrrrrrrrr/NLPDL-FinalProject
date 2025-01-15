from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig
import os
import json
import torch
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import faiss
from sentence_transformers import SentenceTransformer, util
import time
import argparse
import random
random.seed(0)


def load_retrieve_model(finetune=False):
    if finetune:
        # 微调过的
        bert_path = "/ceph/home/yangshu/lema/finetune/checkpoint_0/model_state_2.pt"
        checkpoint = torch.load(bert_path)
        bertconfig = BertConfig.from_pretrained("/ceph/home/yangshu/lema/final_project/mathbert")
        bertmodel = BertModel(bertconfig)
        bertmodel.load_state_dict(checkpoint, strict=False)
        berttokenizer = BertTokenizer.from_pretrained("/ceph/home/yangshu/lema/final_project/mathbert")
        return bertmodel, berttokenizer
    else:
        # 未微调过的
        sentence_model_path = "/ceph/home/yangshu/lema/final_project/mathbert"
        sentence_model = AutoModel.from_pretrained(sentence_model_path)
        sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
        return sentence_model, sentence_tokenizer



# 加载FAISS索引和文档
def load_faiss_index(index_path, documents_path):
    index = faiss.read_index(index_path)
    documents = []
    with open(documents_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            documents.append(data)
    return index, documents

# 检索相关文档
def retrieve_documents(question, index, documents, retrieve_model, retrieve_tokenizer, k=5):
    question_embedding = retrieve_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        question_embedding = retrieve_model(**question_embedding).last_hidden_state.mean(dim=1)
    
    D, I = index.search(question_embedding.numpy(), k)
    retrieved_docs = [documents[i] for i in I[0]]
    

    retrieved_docs = random.choice(retrieved_docs)
    # 再转换成列表
    retrieved_docs = [retrieved_docs]
    
    return retrieved_docs

# 修改模型推理函数，加入检索到的文档作为上下文
def model_inference_rag(question, retrieved_docs, tokenizer, model):
    # 构建包含检索文档的提示

    # correction set
    #retrieved_docs = [f"Problem: {doc['question']}\nWrongAnswer: {doc['llm_answer']}\nCorrectAnswer:{doc['correct_solution']}" for doc in retrieved_docs]

    # training set
    retrieved_docs = [f"Problem:{doc['problem']}\nAnswer:{doc['solution']}" for doc in retrieved_docs]

    context = "\n".join(retrieved_docs)
    prompt = f"Here is an example:\n{context}\nAfter reading the example, please answer the following question.\nQuestion: {question}\nAnswer:"

    # 对问题进行编码
    inputs = tokenizer(prompt, return_tensors="pt")

    # 如果长度超过模型的最大长度，就截断
    if inputs['input_ids'].shape[1] > 2500:
        inputs['input_ids'] = inputs['input_ids'][:, :2500]
        inputs['attention_mask'] = inputs['attention_mask'][:, :2500]
    
    # 使用模型生成答案
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=2800, num_return_sequences=1)
            
        
    # 解码并返回生成的答案
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip(), prompt

# 从文件夹加载数据
def load_data_from_folder(folder_path):
    problems = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                problems.append(data)
    return problems

# 从jsonl文件加载数据
def load_data_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problems.append(data)
    return problems

# 计算准确率
def calculate_accuracy(problems, index, documents, tokenizer, model, output_file, retrieve_model, retrieve_tokenizer):
    correct_count = 0
    total_count = 0

    for problem in problems:
        question = problem['prompt']
        correct_answer = problem['completion']
        test_type = problem['type']
        test_level = problem['level']

        # 提取correct_answer中的答案部分
        if "answer is" in correct_answer.lower():
            correct_answer = correct_answer.split("answer is")[1].split(".")[0].strip()

        t0 = time.time()
        # 检索相关文档
        retrieved_docs = retrieve_documents(question, index, documents, retrieve_model, retrieve_tokenizer)
        doc_type = retrieved_docs[0]['type']
        doc_level = retrieved_docs[0]['level']        

        t1 = time.time()
        print(f"Retrieval time: {t1 - t0:.2f}s")

        # 使用模型生成回答
        predicted_answer_rag, my_prompt = model_inference_rag(question, retrieved_docs, tokenizer, model)
        
        t2 = time.time()
        print(f"Inference time: {t2 - t1:.2f}s")

        # 如果模型中包含“the answer is”，则提取它后面的部分作为生成的答案
        if "answer is" in predicted_answer_rag.lower():
            predicted_answer_rag = predicted_answer_rag.split("answer is")[1].strip()

        # 把生成的答案和真实答案以jsonl形式输出到文件里
        with open(output_file, 'a') as file:
            json.dump({'prompt': my_prompt, 'correct_answer': correct_answer, 'predicted_answer': predicted_answer_rag, 'test_level': test_level, 'doc_level': doc_level, 'test_type': test_type, 'doc_type': doc_type}, file)
            file.write('\n')
        
        # 比较生成的答案与真实答案
        if correct_answer.lower() in predicted_answer_rag.lower():
            correct_count += 1
        
        total_count += 1

    accuracy_rag = correct_count / total_count
    
    return accuracy_rag

def main(folder_path, output_file, index_path, documents_path, finetune, model_path):
    # 加载数据
    problems = load_data_from_jsonl(folder_path)
    
    # 加载FAISS索引和文档
    index, documents = load_faiss_index(index_path, documents_path)
    
    # 加载模型和分词器
    retrieve_model, retrieve_tokenizer = load_retrieve_model(finetune)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 计算准确率
    accuracy_rag = calculate_accuracy(problems, index, documents, tokenizer, model, output_file, retrieve_model, retrieve_tokenizer)

    print(f'Accuracy: {accuracy_rag:.2f}')

if __name__ == '__main__':
            
    # 命令行读入
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, help="The path to the FAISS index.")
    parser.add_argument("--documents_path", type=str, help="The path to the documents.")
    parser.add_argument("--finetune", type=bool, help="Whether to use the fine-tuned model.")
    parser.add_argument("--output_file", type=str, help="The path to the output file.")
    parser.add_argument("--folder_path", type=str, help="The path to the folder containing the test data.")
    parser.add_argument("--model_path", type=str, help="The path to the model.")
    
    
    args = parser.parse_args()

    index_path = args.index_path
    documents_path = args.documents_path
    finetune = args.finetune
    output_file = args.output_file
    folder_path = args.folder_path
    model_path = args.model_path
    
    main(folder_path, output_file, index_path, documents_path, finetune, model_path)

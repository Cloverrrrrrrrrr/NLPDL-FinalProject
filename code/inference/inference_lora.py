from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import time
import argparse

from peft import LoraConfig, TaskType, get_peft_model


def load_data_from_folder(folder_path):
    
    problems = []
    for filename in os.listdir(folder_path):
        
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                problems.append(data)
    return problems

def load_data_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problems.append(data)
    return problems

def load_lora_model():
    lora_path = "autodl-tmp/llama3_lora"
    config = LoraConfig.from_pretrained(lora_path)
    model_path = "/root/autodl-tmp/llama3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    return model, tokenizer


def model_inference_cot(question, model, tokenizer):
    # 在问题后面加上 CoT 提示
    prompt = f"{question}\nAfter you answer the question, please conclude your answer with 'The final answer is:'."
    
    # 对问题进行编码
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 使用模型生成答案
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1500, num_return_sequences=1)
    
    # 解码并返回生成的答案
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()




def calculate_accuracy(problems, output_file, model, tokenizer):    
    correct_count_cot = 0
    
    total_count = 0
    current_count = 0

    for problem in problems:
        
        question = problem['prompt']
        correct_answer = problem['completion']
        test_type = problem['type']
        st = time.time()
        # 提取correct_answer中answer is 后面 句号前面的部分作为真实答案
        if "answer is" in correct_answer.lower():
            correct_answer = correct_answer.split("answer is")[1].split(".")[0].strip()

        # 使用模型生成回答
        predicted_answer_cot = model_inference_cot(question, model, tokenizer)
        
        # 把生成的答案和真实答案以jsonl形式输出到文件里
        with open(output_file, 'a') as file:
            json.dump({'question': question, 'correct_answer': correct_answer, 
            'predicted_answer': predicted_answer_cot,
            'test_type': test_type
            }, file)
            file.write('\n')
        
        if correct_answer.lower() in predicted_answer_cot.lower():
            correct_count_cot += 1

        current_count += 1
        total_count += 1
        print(f"Time: {time.time() - st}")
    
    accuracy_cot = correct_count_cot / current_count
    
    
    return accuracy_cot


def main(folder_path, output_file):
    # 加载数据
    problems = load_data_from_jsonl(folder_path)

    # 加载模型
    model, tokenizer = load_lora_model()

    # 计算正确率
    accuracy_cot = calculate_accuracy(problems, output_file, model, tokenizer)
    
    print(f"Accuracy with CoT: {accuracy_cot:.2f}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    test_path = args.test_path
    output_file = args.output_file
    main(test_path, output_file)


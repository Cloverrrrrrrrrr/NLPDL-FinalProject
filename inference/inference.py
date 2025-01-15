import json
import torch
import time
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from collections import defaultdict
import tqdm

# 设置日志输出到文件
logging.basicConfig(
    filename='model_inference.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def load_data_from_jsonl(jsonl_path):
    """
    从 JSONL 文件加载数据
    """
    problems = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problems.append(data)
    logger.info(f"Loaded {len(problems)} problems from {jsonl_path}")
    return problems

def load_model_and_tokenizer(model_path, lora_path=None):
    """
    加载预训练的模型和tokenizer
    """
    if lora_path is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    else:
        config = LoraConfig.from_pretrained(lora_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Model loaded and moved to {device}")
    
    return model, tokenizer, device


def model_inference_cot(model, tokenizer, device, questions, batch_size=32):
    """
    批量处理问题，进行推理
    """
    predictions = []
    num_batches = len(questions) // batch_size + (1 if len(questions) % batch_size > 0 else 0)
    
    for batch_idx in tqdm.tqdm(range(num_batches), total=num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        prompts = [f"{q}\nAfter you answer the question, please conclude your answer with 'The final answer is:'." for q in batch_questions]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, num_return_sequences=1)

        batch_answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend(batch_answers)

        logger.info(f"Processed batch {batch_idx+1}/{num_batches} (size={len(batch_questions)})")
    
    return predictions


def calculate_accuracy(problems, predicted_answers, output_file):
    """
    计算模型准确率并将结果写入输出文件
    """
    correct_count_cot = 0
    total_count = 0
    task_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    output_data = []

    for problem, predicted_answer in zip(problems, predicted_answers):
        total_count += 1
        question = problem['prompt']
        correct_answer = problem['completion']
        test_type = problem['type']

        # 提取 correct_answer 中 answer is 后面部分作为真实答案
        if "answer is" in correct_answer.lower():
            correct_answer = correct_answer.split("answer is")[1].split(".")[0].strip()

        # 记录问题、正确答案、预测答案、类型等数据
        output_data.append({
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'test_type': test_type,
        })

        # 计算 CoT 答案是否正确
        if correct_answer.lower() in predicted_answer.lower():
            correct_count_cot += 1
            task_accuracy[test_type]['correct'] += 1
        task_accuracy[test_type]['total'] += 1

    # 计算最终准确率
    accuracy_cot = correct_count_cot / total_count if total_count > 0 else 0.0

    # 计算每个 task type 的准确率
    task_accuracies = {task: accuracy['correct'] / accuracy['total'] if accuracy['total'] > 0 else 0.0 
                       for task, accuracy in task_accuracy.items()}

    # 将结果批量写入文件
    with open(output_file, 'w') as file:
        for entry in output_data:
            json.dump(entry, file)
            file.write('\n')

    return accuracy_cot, task_accuracies


def main(args):
    """
    主函数，加载数据、推理和计算准确率
    """
    # 1. 加载数据
    problems = load_data_from_jsonl(args.input_file)

    # 2. 加载模型和tokenizer
    if args.use_lora:
        model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.lora_path)
    else:
        model, tokenizer, device = load_model_and_tokenizer(args.model_path)

    # 3. 提取问题并进行批量推理
    questions = [problem['prompt'] for problem in problems]
    st = time.time()
    predicted_answers = model_inference_cot(model, tokenizer, device, questions, args.batch_size)
    inference_time = time.time() - st
    logger.info(f"Inference completed in {inference_time:.2f} seconds")

    # 4. 计算准确率并保存结果
    accuracy_cot, task_accuracies = calculate_accuracy(problems, predicted_answers, args.output_file)
    
    logger.info(f"Accuracy with CoT: {accuracy_cot:.2f}")
    logger.info("Task-wise accuracies:")
    for task, accuracy in task_accuracies.items():
        logger.info(f"  {task}: {accuracy:.2f}")

    return accuracy_cot, task_accuracies


if __name__ == "__main__":
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="Run inference with the model.")
    
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input jsonl file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file to save results.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--use_lora', action='store_true', help="Use LoRA adapter for inference.")
    parser.add_argument('--lora_path', type=str, help="Path to the LoRA model directory.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference.")
    
    # 2. 解析命令行参数
    args = parser.parse_args()
    if args.use_lora and not args.lora_path:
        raise ValueError("LoRA path must be provided when using LoRA adapter.")
    
    # 3. 调用主函数
    main(args)

import json
import argparse 

def is_correct(correct_answer, predicted_answer, output_path):
    # 忽略大小写
    correct_answer = correct_answer.lower()
    predicted_answer = predicted_answer.lower()

    # 把predicted_answer里面Let's think step by step.之前的文本都删掉
    if "let's think step by step." in predicted_answer:
        predicted_answer = predicted_answer.split("let's think step by step.")[1]

    else:
        print("let's think step by step. not found")
        return False

    if "answer is" in predicted_answer:
        predicted_answer = predicted_answer.split("answer is")[1]
        if "." in predicted_answer:
            predicted_answer = predicted_answer.split(".")[0]
    else:
        # 如果格式不对，取最后的一部分字符
        predicted_answer = predicted_answer[-100:] 

    return correct_answer in predicted_answer

def calculate_accuracy(jsonl_file, output_path):
    total = 0
    correct = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            correct_answer = data.get('correct_answer', '')
            predicted_answer = data.get('predicted_answer', '')

            if is_correct(correct_answer, predicted_answer, output_path):
                correct += 1
            
            total += 1
    
    # 计算正确率
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total

# 读取jsonl文件统计每一行里'test_type'和'doc_type'相同的比例
def calculate_accuracy2(jsonl_file):
    total = 0
    correct = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            test_type = data['test_level']
            doc_type = data['doc_level']
            
            # 判断当前回答是否正确
            if test_type == doc_type:
                correct += 1
            
            total += 1
    
    # 计算正确率
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f'Accuracy: {accuracy:.2f}% ({correct}/{total})')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    output_path = args.output_path
    
    accuracy, correct_count, total_count = calculate_accuracy('/ceph/home/yangshu/lema/final_project/project/rag_error.jsonl', output_path)
    print(f'Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})')
    
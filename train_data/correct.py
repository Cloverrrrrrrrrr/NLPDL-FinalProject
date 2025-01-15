# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
from openai import OpenAI
import tqdm
from prompt import JUDGE_PROMPT
import os

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1"
)

def judge(pred: str, label: str) -> bool:
    """
    use the deepseek api to judge the correctness of the prediction.
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Your are a helpful assistant."},
            {"role": "user", "content": JUDGE_PROMPT.format(label, pred)},
        ],
        stream=False
    )

    if any(keyword in response.choices[0].message.content for keyword in ['true', 'True', 'TRUE']):
        return True
    else:
        return False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path to gold file and inference file.')
    parser.add_argument('--groundtruth_file', type=str, required=True)
    parser.add_argument('--prediction_file', type=str, required=True)
    parser.add_argument('--error_file', type=str, required=True)
    parser.add_argument('--correct_file', type=str, required=True)
    args = parser.parse_args()

    file_label = args.groundtruth_file
    file_pred = args.prediction_file

    print('file_label:', file_label)
    print('file_pred:', file_pred)

    acc_list = []
    with open(file_label, 'r', encoding='utf-8') as f_label, \
            open(file_pred, 'r', encoding='utf-8') as f_pred:
        label_infos = [json.loads(line) for line in f_label.readlines()]
        pred_infos = [json.loads(line) for line in f_pred.readlines()]

        assert len(label_infos) == len(pred_infos)
        
        for label_info, pred_info in tqdm.tqdm(zip(label_infos, pred_infos), total=len(label_infos)):
            pred = pred_info['llm_answer']
            label = label_info['solution']
            
            if judge(pred, label):
                with open(args.correct_file, 'a', encoding='utf-8') as f_correct:
                    f_correct.write(json.dumps(pred_info) + '\n')
                acc_list.append(1)
            else:
                pred_info.update({"correct_solution": label})
                with open(args.error_file, 'a', encoding='utf-8') as f_error:
                    f_error.write(json.dumps(pred_info) + '\n')
                acc_list.append(0)

    print('acc:', sum(acc_list) / len(acc_list))


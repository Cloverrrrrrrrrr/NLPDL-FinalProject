# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging, json
import argparse
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def llm_inference(testdata_file, output_file, model_path):
    # log args
    logger.info(f"test file: {testdata_file}")
    logger.info(f"output file: {output_file}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(testdata_file, 'r', encoding='utf-8') as f_read:
        test_questions = [json.loads(line)['problem'] for line in f_read.readlines()]
    test_prompts = [
        [{"role": "user", "content": question + "Let's think step by step."}] for question in test_questions
    ]
    prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in test_prompts]
    total_lines = len(prompt_token_ids)
    logger.info(f"Total lines: {total_lines}")
    assert len(test_prompts) != 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024
    }
    for i in tqdm.tqdm(range(total_lines), total=total_lines):
        input_ids = torch.tensor(prompt_token_ids[i]).unsqueeze(0).to(device)
        outputs = model.generate(input_ids, **sampling_params)
        sample = {
            "question": test_questions[i],
            "llm_answer": tokenizer.decode(outputs[0], skip_special_tokens=True),
        }
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for running generation.')
    parser.add_argument('--testdata_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    
    llm_inference(testdata_file=args.testdata_file,
              output_file=args.output_file,
              model_path=args.model_path)
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging, json
import argparse
from prompt import Q_PROMPT
import tqdm
from openai import OpenAI
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
            
def inference(testdata_file, output_file, model_name):
    
    # log args
    logger.info(f"test file: {testdata_file}")
    logger.info(f"output file: {output_file}")

    with open(testdata_file, 'r', encoding='utf-8') as f_read:
        test_questions = [json.loads(line)['problem'] for line in f_read.readlines()]
        messages = [
            [
                {"role": "system", "content": Q_PROMPT},
                {"role": "user", "content": question}
            ] 
            for question in test_questions
        ]
    assert len(messages) != 0

    for i, message in tqdm.tqdm(enumerate(messages), total=len(messages)):
        outputs = client.chat.completions.create(
            model=model_name,
            messages=message,
        )
        output = {"question": test_questions[i], "llm_answer": outputs.choices[0].message.content}
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(output) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for running generation.')
    parser.add_argument('--testdata_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    inference(testdata_file=args.testdata_file,
              output_file=args.output_file,
              model_path=args.model_name)
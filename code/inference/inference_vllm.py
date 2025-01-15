import logging
import os
import json
from vllm import LLM, SamplingParams
import argparse
from prompt import Q_PROMPT
import tqdm
import torch
import gc

def inference(testdata_file, output_file, model_path, tensor_parallel_size, trust_remote_code):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # log args
    logger.info(f"test file: {testdata_file}")
    logger.info(f"output file: {output_file}")
    logger.info(f"tensor_parallel_size: {tensor_parallel_size}")

    with open(testdata_file, 'r', encoding='utf-8') as f_read:
        test_data = [json.loads(line) for line in f_read.readlines()]
        # 在test_prompt最后加上，模型输出完毕后要加上一句话，The answer is，然后再加上模型输出
        test_prompts = []
        for item in test_data:
            prompt = item['prompt']
            prompt += ' After you answer the question, please conclude your answer with "The final answer is" and then provide your answer.'
            test_prompts.append(prompt)
        
        # 在completion里提取 The answer is 后面 句号前面的部分

        test_completions = []
        for item in test_data:
            completion = item['completion']
            if 'The answer is' in completion:
                completion = completion.split('The answer is ')[1].split('.')[0].strip()
            test_completions.append(completion)
        #test_completions = [item['completion'] for item in test_data]

        test_types = [item['type'] for item in test_data]
        total_lines = len(test_prompts)
        logger.info(f"Total lines: {total_lines}")
    
    assert len(test_prompts) != 0

    llm = LLM(model=model_path,
              tensor_parallel_size=tensor_parallel_size,
              trust_remote_code=trust_remote_code,
              max_num_batched_tokens=8192,
              max_model_len=8192,
              gpu_memory_utilization=0.8)

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048)

    batch_size = 64
    total_batch_num = (total_lines // batch_size) + 1
    print(total_batch_num)

    current_lines = 0
    all_outputs = []
    correct_answers = 0

    try:
        for batch_idx in tqdm.tqdm(range(total_batch_num), total=total_batch_num):
            print("batch start")
            if batch_idx == total_batch_num - 1:
                prompt_batch = test_prompts[batch_idx * batch_size:]
                completion_batch = test_completions[batch_idx * batch_size:]
                type_batch = test_types[batch_idx * batch_size:]
            else:
                prompt_batch = test_prompts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                completion_batch = test_completions[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                type_batch = test_types[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            results = llm.generate(prompt_batch, sampling_params)
           
            #results = [None] * batch_size
            current_lines += batch_size
            logger.info(f"{current_lines} in {total_lines} examples.")

            
            
            for i, result in enumerate(results):
                # 检查 result.outputs 是否有内容
                if len(result.outputs) > 0:
                    generated_answer = result.outputs[0].text.strip()
                else:
                    logger.error(f"No outputs found for prompt: {prompt_batch[i]}")
                    generated_answer = ""  # 或者设置为其他合适的默认值

                old_generated_answer = generated_answer

                # 如果生成的答案包含 "The answer is"，提取后面的部分
                if 'The answer is' in generated_answer:
                    # 使用 split 获取答案，避免索引越界
                    parts = generated_answer.split('The answer is ')
                    if len(parts) > 1:
                        generated_answer = parts[1].split('.')[0].strip()
                    else:
                        generated_answer = parts[0].strip()
                else:
                    # 如果没有 "The answer is"，直接取最后一句话
                    generated_answer = generated_answer.split('.')[-1].strip()

                # 检查 completion_batch 是否足够长
                if i < len(completion_batch):
                    correct_answer = completion_batch[i].strip()
                else:
                    logger.error(f"Index {i} out of range for completion_batch.")
                    correct_answer = ""  # 或者设置为其他合适的默认值

                # Compare the model output with the correct answer
                if correct_answer in generated_answer:
                    correct_answers += 1

                all_outputs.append(
                    {
                        'prompt': prompt_batch[i],
                        'generated_answer': old_generated_answer,
                        'correct_answer': correct_answer,
                        'test_type': type_batch[i],
                    }
                )
            accuracy = correct_answers / total_lines
            logger.info(f"Accuracy: {accuracy * 100:.2f}%")

            # Save the output
            with open(output_file, "w", encoding='utf-8') as f:
                for output in all_outputs:
                    f.write(json.dumps(output) + '\n')
             

        # Calculate accuracy
        accuracy = correct_answers / total_lines
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")

        # Save the output
        with open(output_file, "w", encoding='utf-8') as f:
            for output in all_outputs:
                f.write(json.dumps(output) + '\n')

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Cleaning up...")
        
        # Release GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared.")

        # Optionally, you can forcefully shut down the model (if needed):
        #llm.close()
        logger.info("Model closed.")

        # Handle any other cleanup tasks
        logger.info("Exiting gracefully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared after error.")
        #llm.close()
        logger.info("Model closed after error.")

    finally:
        logger.info("Inference process finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--testdata_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--trust_remote_code", type=bool, default=False)

    args = parser.parse_args()

    testdata_file = args.testdata_file
    output_file = args.output_file
    model_path = args.model_path
    tensor_parallel_size = args.tensor_parallel_size
    trust_remote_code = args.trust_remote

    inference(testdata_file=testdata_file,
                output_file=output_file,
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code)

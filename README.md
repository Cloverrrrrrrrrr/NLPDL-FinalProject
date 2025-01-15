# Learning from Mistakes

## Dependencies
For **Data Generation** and **Contrastive Finetuning**, please refer to finetuning.yaml to prepare the environment. As for **Error-RAG**, please install the dependencies by running the following command:
```bash
pip install -r rag/requirements.txt
```

## Data Generation
Generate answers using open-source LLMs:
```
python train_data/opensource.py --testdata_file train_data/data/math.jsonl \
    --output_file train_data/llama.jsonl \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

Generate answers using Qwen2.5-72B:
```
export DASHSCOPE_API_KEY=<YOUR API KEY>

python train_data/openai.py \
    --testdata_file train_data/data/math.jsonl \
    --output_file train_data/qwen2.5-72b-instruct.jsonl \
    --model_name qwen2.5-72b-instruct 
```

Judge the correctness of the answers:
```
export DEEPSEEK_API_KEY=<YOUR API KEY>

python train_data/correct.py \
    --groundtruth_file train_data/data/math.jsonl \
    --prediction_file train_data/llama.jsonl \
    --error_file  train_data/data/llama_error.jsonl \
    --correct_file train_data/data/llama_right.jsonl 
```
## Contrastive Fine-Tuning
### Full Parameter Fine-tuning
modify the training arguments in contrastive/configs/full.yaml:
```
model_name: meta-llama/Meta-Llama-3-8B-Instruct
data_path: train_data/data/llama_error.jsonl
output_dir: contrastive/full
learning_rate: 0.00001
per_device_train_batch_size: 1
num_train_epochs: 2
weight_decay: 0.01
optim: adamw_torch_4bit
fp16: False
bf16: True
gradient_accumulation_steps: 4
use_lora: False
```
then run the following command to start training:
```
python contrastive/train.py --config contrastive/configs/full.yaml
```
### LoRA
modify the training arguments in contrastive/configs/lora.yaml:
```
model_name: meta-llama/Meta-Llama-3-8B-Instruct
data_path: train_data/data/llama_error.jsonl
output_dir: contrastive/lora
learning_rate: 0.00001
per_device_train_batch_size: 3
um_train_epochs: 2
weight_decay: 0.01
optim: adamw_torch
fp16: False
bf16: True
gradient_accumulation_steps: 4
use_lora: True
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
```
then run the following command to start training:
```
python contrastive/train.py --config contrastive/configs/lora.yaml
```

## Error-RAG

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

## Inference
use the following command for running batch inference and calculate the accuracies for MATH dataset:
```
python inference/inference.py --input_file test_data/math_test.jsonl \
    --output_file full.jsonl \
    --model_path contrastive/full/checkpoint \
    --batch_size 64
```
if using LoRA adapter:
```
python inference/inference.py --input_file test_data/math_test.jsonl \
    --output_file full.jsonl \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --use_lora \
    --lora_path contrastive/lora/checkpoint \
    --batch_size 64
```
To run inference with RAG:
```
python inference/inference_rag_new.py \
    --index_path yourindex.index \
    --documents_path data/documents.json \
    --finetune False \
    --output_file your_output_path.jsonl \
    --folder_path test_data/math_test.jsonl \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct
```
To create the index of document embeddings using faiss:
```
python rag/faiss/creatingfaiss.py
    --model_path your_model_path\
    --state_dict_path your_statedict_path\
    --output_file your_output_file\
    --document_path your_document_path.jsonl
```
To calculate the accuracy from a .jsonl file:
```
python results/count_accuracy.py\
    --input_path your_input_path.jsonl
```
To create PCA images from faiss index and its document:
```
python results/pca.py\
    --index_path your_index_path.index\
    --documents_path your_documents_path.jsonl\
    --output_file your_output_file.png
```
To finetune BERT model using triple loss:
```
python rag/finetune_bert/finetune.py\
    --query_data your_query_data_path.jsonl\
    --doc_data your_doc_data.jsonl\
    --model_path tbs17/MathBERT\
    --output_dir your_output_dir
```
To finetune BERT model using Cluster loss:
```
python rag/finetune_bert/finetune2.py\
    --data_path your_data_path.jsonl\
    --model_path tbs17/MathBERT\
    --save_path your_output_dir
```

    

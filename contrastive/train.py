from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
import torch.nn.functional as F
import argparse
import torch
import yaml
from peft import get_peft_model, LoraConfig, TaskType

# prepare data, use question as anchor, correct answer as positive, incorrect answer as negative
class TripletDataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        question = self.raw_data[idx]['question']
        correct = self.raw_data[idx]['correct_solution']
        incorrect = self.raw_data[idx]['llm_answer']

        return question, correct, incorrect
    
def collate_fn(batch, tokenizer):
    questions, corrects, incorrects = zip(*batch)
    tokenizer.pad_token = tokenizer.eos_token
    question_encodings = tokenizer(questions, padding="longest", truncation=True, return_tensors='pt')
    correct_encodings = tokenizer(corrects, padding="longest", truncation=True, return_tensors='pt')
    incorrect_encodings = tokenizer(incorrects, padding="longest", truncation=True, return_tensors='pt')
    
    return question_encodings, correct_encodings, incorrect_encodings

def triplet_loss(anchor, positive, negative, margin):
    # calculate cosine similarity between anchor and positive, anchor and negative
    positive_similarity = F.cosine_similarity(anchor, positive)
    negative_similarity = F.cosine_similarity(anchor, negative)
    
    # calculate loss
    loss = F.relu(negative_similarity - positive_similarity + margin).mean()
    return loss

class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        question, positive_input, negative_input = inputs
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        question = {k: v.to(device) for k, v in question.items()}
        positive_input = {k: v.to(device) for k, v in positive_input.items()}
        negative_input = {k: v.to(device) for k, v in negative_input.items()}
        
        # Model outputs
        question_output = model(**question, output_hidden_states=True) 
        question_embedding = question_output.hidden_states[-1].mean(dim=1)  # mean pooling

        positive_output = model(**positive_input, output_hidden_states=True)
        positive_embedding = positive_output.hidden_states[-1].mean(dim=1)

        negative_output = model(**negative_input, output_hidden_states=True)
        negative_embedding = negative_output.hidden_states[-1].mean(dim=1)
        
        # Calculate triplet loss
        loss = triplet_loss(question_embedding, positive_embedding, negative_embedding, margin=0.2)
        
        return loss if not return_outputs else (loss, question_embedding, positive_embedding, negative_embedding)
    
def train(config: dict):
    
    # load model and tokenizer
    if config["bf16"]:
        model = AutoModelForCausalLM.from_pretrained(config['model_name'], device_map="auto", torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    elif config["fp16"]:
        model = AutoModelForCausalLM.from_pretrained(config['model_name'], device_map="auto", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(config['model_name'], device_map="auto").to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    if config["use_lora"]:
        # Apply LoRA adapter
        lora_config = LoraConfig(
            r=config["lora_r"], 
            inference_mode=False,
            lora_alpha=config["lora_alpha"],  # scaling factor for LoRA
            lora_dropout=config["lora_dropout"],  # dropout for LoRA layers
            task_type=TaskType.CAUSAL_LM,  # specify the task type (causal LM)
        )
        model = get_peft_model(model, lora_config)

    # load data from jsonl file
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # create dataset
    dataset = TripletDataset(data, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        learning_rate=config['learning_rate'],  
        per_device_train_batch_size=config['per_device_train_batch_size'],
        num_train_epochs=config['num_train_epochs'],
        weight_decay=config['weight_decay'],
        torch_empty_cache_steps=config["gradient_accumulation_steps"],
        max_grad_norm=1.0,
        optim=config['optim'],
        overwrite_output_dir=True,
        fp16=config['fp16'],
        bf16=config['bf16'],
        warmup_steps=10,
        logging_dir='./logs',  
        save_strategy="epoch", 
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        report_to="wandb"
    )
        
    trainer = TripletTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=lambda data: collate_fn(data, tokenizer)
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    train(config)

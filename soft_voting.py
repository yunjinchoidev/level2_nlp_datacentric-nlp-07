from tqdm import tqdm
import os
import random
import numpy as np
import pandas as pd
import datetime
from pytz import timezone
import wandb

import torch
from torch.utils.data import Dataset, DataLoader

import evaluate
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold

### custom ###
N_SPLITS = 5
N_ENSEMBLE = 3
EVAL_BATCH_SIZE = 1024
model_name = "klue/roberta-large"
data = pd.read_csv("../data/train_final.csv")
###

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained(model_name)

data = data.drop(data[data["text"].isnull() == True].index)
raw_texts, labels = data["text"].values, data["target"].values
num_classes = len(set(labels))

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        self.ids = []
        for text, label in tqdm(zip(input_texts, targets), total=len(input_texts)):
            tokenized_input = tokenizer(text, max_length=40,padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)
    
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_strategy='no',
    evaluation_strategy='epoch',
    save_strategy='no',
    logging_steps=100,
    eval_steps=500,
    save_steps=100,
    save_total_limit=2,
    learning_rate= 2e-05,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon=1e-08,
    weight_decay=0.01,
    warmup_ratio=0.6,
    lr_scheduler_type='linear',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    load_best_model_at_end=False,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    # seed=SEED,
    report_to='wandb',
    fp16=True
)

model_probs = [[] for _ in range(N_ENSEMBLE)]
pred_probs = []
indices = []
skfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True) # 5 -> valid 0.2

now = datetime.datetime.now(timezone("Asia/Seoul")).strftime("%m/%d %H:%M")

for fold_idx, (used_index, test_index) in enumerate(skfold.split(raw_texts,labels)):
    print(f"*** prepare test_dataset for fold {fold_idx+1}/{N_SPLITS} ***")
    test_dataset = BERTDataset(data.iloc[test_index], tokenizer)

    if N_ENSEMBLE == 1:
        model_idx = 0
        train_dataset = BERTDataset(data.iloc[used_index], tokenizer)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        
        model.eval()
        for batch in tqdm(DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE), desc=f"evaluate{fold_idx+1}/{N_SPLITS}"):
            inputs = {
                'input_ids':batch['input_ids'].to(DEVICE),
                'attention_mask':batch['attention_mask'].to(DEVICE),
            }
            label = batch['labels']
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.Softmax(dim=1)(logits).cpu().numpy()
                model_probs[model_idx].append(probs)
                # pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                # preds.extend(pred)
                
    else: 
        used_label = [labels[i] for i in used_index]
    
        ## ensemble for test_index
        skfold2 = StratifiedKFold(n_splits=N_ENSEMBLE, shuffle=True)

        for model_idx, (train_index, valid_index) in enumerate(skfold2.split(used_label, used_label)):
            train_index = [used_index[i] for i in train_index]
            valid_index = [used_index[i] for i in valid_index]

            print(f"*** prepare train, valid dataset for fold {fold_idx+1}/{N_SPLITS} model {model_idx+1}/{N_ENSEMBLE} ***")
            train_dataset = BERTDataset(data.iloc[train_index], tokenizer)
            valid_dataset = BERTDataset(data.iloc[valid_index], tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()

            model.eval()
            for batch in tqdm(DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE), desc=f"evaluate{fold_idx+1}/{N_SPLITS}"):
                inputs = {
                    'input_ids':batch['input_ids'].to(DEVICE),
                    'attention_mask':batch['attention_mask'].to(DEVICE),
                }
                label = batch['labels']
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.Softmax(dim=1)(logits).cpu().numpy()
                    model_probs[model_idx].append(probs)
                    # pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                    # preds.extend(pred)
        
    indices.extend(test_index)       

for i in range(N_ENSEMBLE):
    model_probs[i] = np.concatenate(model_probs[i])
    
def vote(probs):
    res = probs[0].copy()
    for i in range(1, len(probs)):
        res += probs[i]
    
    return res / len(probs)

def probs_to_string(probs):
    res = []
    for ps in probs:
        s=""
        for p in ps:
            s+=str(p)+" "
        res.append(s)
    return res

def get_preds(probs):
    res = []
    for p in probs:
        res.append(np.argmax(p))
    return res

### warning! need to customize ###
mean_probs = vote(model_probs)
res_df = pd.DataFrame({
    "text": data.iloc[indices]["text"],
    "probs_0": probs_to_string(model_probs[0]),
    "probs_1": probs_to_string(model_probs[1]),
    "probs_2": probs_to_string(model_probs[2]),
    "probs_m" : probs_to_string(mean_probs),
    "pred_0": get_preds(model_probs[0]),
    "pred_1": get_preds(model_probs[1]),
    "pred_2": get_preds(model_probs[2]),
    "pred_m": get_preds(mean_probs)    
})
res_df.to_csv("roberta_ensemble.csv")
###
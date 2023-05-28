import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

model = GPT2LMHeadModel.from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5')
tokenizer = AutoTokenizer.from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5')

train = pd.read_csv('./train.csv')
concats = []
for label, text in zip(train['label_text'] ,train['input_text']):
    concats.append(label+'.'+text)
    
    
class GPTDataset(Dataset):
    def __init__(self, tokenizer, concats):
        self.item = tokenizer(concats,return_tensors = 'pt', padding=True, max_length=128)['input_ids']
        self.length = len(concats)

    def __getitem__(self, i):
        return self.item[i]

    def __len__(self):
        return self.length
    

if __name__=="__main__":
    dataset = GPTDataset(tokenizer, concats)
    batch_size = 16
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    epochs = 10
    lr = 2e-5
    acc_steps = 100
    warmup_steps = 200
    device = torch.device("cuda")
    model = model.cuda()
    model.train()
    save_model_on_epoch = True
    output_dir = '.'
    output_prefix= "finetune_test"

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)

    loss = 0

    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        print(loss)
        for input_text in tqdm(train_dataloader):
            input_tensor = input_text.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()


            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

            input_tensor = None
        if save_model_on_epoch:
            torch.save(model, f"{output_prefix}-{epoch}-{outputs[0].item():0.2f}.pt")
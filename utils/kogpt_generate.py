from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
from hanspell import spell_checker
from tqdm import tqdm
import pandas as pd

def generate(input_text, tokenizer, model, num):
    sentence_list = []
    token_ids = tokenizer(input_text + '|', return_tensors = 'pt')['input_ids'].to(device)
    for _ in tqdm(range(num)):
        gen_ids = model.generate(token_ids, max_length=32,
                                 repetition_penalty=2.0,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 bos_token_id=tokenizer.bos_token_id,
                                 use_cache=True,
                                 # temperature=2.0,
                                 do_sample=True,
                                )
        sentence = tokenizer.decode(gen_ids[0])
        sentence = sentence[sentence.index('|')+1:]
        if '<pad>' in sentence:
            sentence = sentence[:sentence.index('<pad>')].rstrip()
        sentence = sentence.replace('<unk>', ' ').split('\n')[0]
        # sentence = spell_checker.check(sentence).checked
        sentence_list.append(sentence)
    return sentence_list

if __name__=="__main__":
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained('JLake310/ko-gpt-trinity-1.2B-ynat-gen').to(device)
    tokenizer = AutoTokenizer.from_pretrained('JLake310/ko-gpt-trinity-1.2B-ynat-gen')
    
    labels = ['정치', '경제', '사회', '생활문화', '세계', 'IT과학', '스포츠']
    
    gen_sentences = {}
    gen_num = 10000
    
    for label in labels:
        print(f'Generating {label}...')
        gen_sentences[label] = generate(label, tokenizer, model, gen_num)
    
    label_nums = []
    for num in range(7):
        nums = [num] * gen_num
        label_nums += nums

    texts = []
    for label in labels:
        texts += gen_sentences[label]
                                        
    gen_data = pd.DataFrame({'target':label_nums, 'text':texts})
    gen_data.to_csv('./gen_data.csv', index=False)
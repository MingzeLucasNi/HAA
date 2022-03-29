import torch
import os
from transformers import (BertForMaskedLM,
                        BertTokenizer,
                        TrainingArguments,
                        Trainer,
                        DataCollatorForLanguageModeling,
                        LineByLineTextDataset)
# import matplotlib.pyplot as plt #画图用的 可以暂时不用。
import numpy as np
from datasets import load_dataset
data_path='/data/lni/attention_attack/bert_model_hate_speech/hate_offensive_data.txt'
check_path='/data/lni/attention_attack/bert_model_hate_speech/checkpoints'
args_path='/data/lni/attention_attack/bert_model_hate_speech/args'
model_path='/data/lni/attention_attack/bert_model_hate_speech/model'

#preprocessing the datasets
data=[]
with open (data_path, 'rt') as file:
    for ele in file:
        data.append(ele.strip())
#data=data[0:2500]
#prepare pretrained bert model and tokenizer
bert=BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
# tokenized_datasets = datasets.map(tokenizer, batched=True, num_proc=4, remove_columns=["text"])

for i in data[0:5]:
	print(i)
#prepare data loader and datasets
data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=True,mlm_probability=0.20)
print('finish datacollator')


dataset = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path=data_path,
    block_size=128)
print('finish datasets')
#set the hyperparameters
args=TrainingArguments(
    output_dir=args_path,
    do_train=True,
    per_device_train_batch_size=32,
    learning_rate=1e-5,
    save_total_limit=15,
    num_train_epochs=60
)

#set up the trainer
trainer=Trainer(
    model=bert,
    args=args,
    data_collator=data_collator,
    train_dataset=dataset,
)

#training model
print('s#############tart training#################')
trainer.train()
#save model
trainer.save_model(model_path)
print('finish traininig!!!!!! Good job!')

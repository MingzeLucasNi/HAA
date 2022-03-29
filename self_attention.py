#bert attention load
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import torch



model_path='/Users/lucas/Desktop/final file/model/fintune-bert'
bert=BertForMaskedLM.from_pretrained(model_path)
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
def bert_attention(text):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(text.split())
    input=tokenizer(text,return_tensors='pt')
    output=bert(**input, output_attentions=True)
    # print(output[1][0].shape)
    attention_score=torch.mean(output[1][0][0],dim=0)
    temps=input['input_ids'][0]
    tokens=tokenizer.convert_ids_to_tokens(temps)
    return attention_score,tokens

for i in range(len(data)):
    if i%10==0:
        print('#####################'+str((i+1)/len(data))+'###########################')
    atts,tokens=bert_attention(data[i])
    attentions.append(atts)
    if len(data[i].split())+2==len(tokens):
        count=count+1
    else:
        print('wrong')
        wrong_index.append(i)

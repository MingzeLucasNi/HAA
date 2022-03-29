import torch
import numpy as np
from transformers import pipeline
from utilities.semantic_scores import *
from utilities.bleu_score import *
from utilities.hug_sem import *
from random import randint
import random
random.seed(10)
bert_path='/Users/lucas/Desktop/attacking nmt/pretrained_models/hate_speech_bert'
unmasker=pipeline('fill-mask', bert_path)
unmasker2=pipeline('fill-mask',"bert-base-cased")
# unmasker2('I like eating [MASK].')
from transformers import BertTokenizer, BertForMaskedLM

tok = BertTokenizer.from_pretrained("bert-base-cased")
bert0 = BertForMaskedLM.from_pretrained("bert-base-cased")

def Word_Marker(attention):
    """
    Usage: Scoring the importance of the words' importance from the attention matrix.
    Input:[tensor] Attention matrix
    Output:[tensor] Scores of the word, excluding the special tokens(<start>,<end>,[CLS])

    """
    atts=torch.sum(attention,dim=0)
    # print(atts)
    key_position_index=torch.sort(atts[1:(len(atts)-1)],dim=0).indices#从不重要到重要
    token_marks_index=torch.sort(key_position_index).indices+1
    return token_marks_index

def combined_key_words(google_attention,bert_attention,bert_weight):
    """
    Usage: Merge two types of attentions together and provide the most
            important word
    Input: google attention,bert_attention, bert_weight
    Output: the index of the most important words

    """

    google_mark=Word_Marker(google_attention)
    bert_mark=Word_Marker(bert_attention)
    # print(google_mark)
    # print(bert_mark)
    combined_mark=bert_weight*bert_mark+(1-bert_weight)*google_mark
    # print(combined_mark)
    one_key_word=torch.max(combined_mark,dim=0).indices
    return one_key_word

# def multiply_attention_adversary(text, bert_attention, google_attention):
#     tokens=text.split()
#     matrix=google_attention@bert_attention
#     values=torch.max(matrix,dim=0).values
#     key_word=torch.max(values,dim=0).indices
#     target_word_str=tokens[key_word]
#     tokens[key_word]='[MASK]'
#     masked_text=' '.join(tokens)
#     adversarial_text=unmasker(masked_text)[0]['sequence']
#     ad_word=unmasker(masked_text)[0]['token_str']
#
#     return adversarial_text,target_word_str,ad_word

def merge_attention_adversary(text,bert_attention,google_attention,bert_weight):
    tokens=text.split()
    key_word=combined_key_words(google_attention,bert_attention,bert_weight)
    target_word_str=tokens[key_word]
    tokens[key_word]='[MASK]'
    masked_text=' '.join(tokens)
    adversarial_text=unmasker(masked_text)[0]['sequence']
    ad_word=unmasker(masked_text)[0]['token_str']

    return adversarial_text,target_word_str,ad_word

#########################################################


def mul_combined_key_words(google_attention,bert_attention,bert_weight):
    """
    Usage: Merge two types of attentions together and provide the most
            important word
    Input: google attention,bert_attention, bert_weight
    Output: the index of the most important words

    """

    google_mark=Word_Marker(google_attention)
    bert_mark=Word_Marker(bert_attention)
    # print(google_mark)
    # print(bert_mark)
    combined_mark=bert_weight*bert_mark+(1-bert_weight)*google_mark
    # print(combined_mark)
    one_key_word=torch.sort(combined_mark,dim=0,descending=True).indices
    return one_key_word
# s='I like eating   .'
# s.split()
def mul_merge_attention_adversary(text,bert_attention,google_attention,bert_weight,N):
    tokens=text.split()
    key_indices=mul_combined_key_words(google_attention,bert_attention,bert_weight)[0:N]
    targets=[]
    subs=[]
    ad_text=text
    # print(key_indices)
    key_indices=torch.sort(key_indices,descending=True).values.tolist()

    for ele in key_indices:
        tokens=ad_text.split()
        # print('index',ele)
        # print('lenngth of tokens',len(tokens))
        # print('tokens:',tokens)
        target_word_str=tokens[ele]
        tokens[ele]='[MASK]'
        # print(tokens)
        masked_text=' '.join(tokens)
        last_score=0
        for i in range(5):
            ad_text=unmasker(masked_text)[i]['sequence']
            ad_word=unmasker(masked_text)[i]['token_str']
            use_scores=semantic_scores(ad_text,text)
            if ad_word!=target_word_str and use_scores>original:
                adversarial_text=ad_text
                ad_words=ad_word
                last_score=use_scores
        targets.append(target_word_str)
        subs.append(ad_words)
    return adversarial_text,targets,subs

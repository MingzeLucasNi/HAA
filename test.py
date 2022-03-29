import torch
import numpy as np
from transformers import pipeline
from bleu_score import *
import random
from self_attention import *
from luong_attention import *

ori_source_text=''
ori_target_text=''

result, sentence, translation_attention=evaluate(ori_source_text)
translation_attention=translation_attention[:len(result.split(' ')), :len(sentence.split(' '))]
self_attention,tokens=bert_attention(ori_source_text)
perturbation_per=0.15
N=float(int(perturbation_per*len(ori_source_text.split())))
lambd=0.42
ad_text=mul_merge_attention_adversary(ori_source_text,self_attention,translaion_attention,bert_weight,N)
print(ad_text)

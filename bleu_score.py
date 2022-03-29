# import packages
import re
import jieba

from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4

#function to merge all the space
def remove_space(text):
  text=text.replace(' ','')
  return text
#function to split chinese senteces with jieba dictionary.
def jieba_split(text):
  test= jieba.cut(text, cut_all=False)
  return " ".join(test)
# function to preprocessing the chinese sentences before calculating BLUE score
def zh_preprocess(text):
  text=jieba_split(remove_space(text))
  return text


# function to calculate the bleu score
def BLEU(cand, ref):
  cand=zh_preprocess(cand).split()
  ref=zh_preprocess(ref).split()

  score=bleu([ref], cand, smoothing_function=smoothie)
  return score

from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
cos = torch.nn.CosineSimilarity(dim=0,eps=1e-6)

def hug_sim(ref, candi):
    pair=[ref,candi]
    encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sim=cos(sentence_embeddings[0], sentence_embeddings[1])
    return sim

# hug_sim('i like eating cakes','I [UN] eating homemade cakes')
# sentences = ['I am Chinese','I am not Chinese']
#
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#
# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#
# # Perform pooling. In this case, max pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#
# print("Sentence embeddings:")
# # sentence_embeddings[1]
# print(torch.max(sentence_embeddings[0,:]))
# torch.randn(100, 128).shape
# cos = torch.nn.CosineSimilarity(dim=0,eps=1e-6)
# output = cos(sentence_embeddings[0], sentence_embeddings[1])
# output
# p = torch.nn.functional.softmax(sentence_embeddings, dim=1)
# torch.max(p)
# from transformers import AutoTokenizer, AutoModel
# import torch
#
#
# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#
#
# # Sentences we want sentence embeddings for
# sentences = ['This is an example sentence', 'Each sentence is converted']
#
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
# model = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
#
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#
# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#
# # Perform pooling. In this case, max pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#
# print("Sentence embeddings:")
# print(sentence_embeddings)

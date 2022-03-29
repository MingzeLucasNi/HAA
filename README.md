# Code for "Hybrid Attentive Attacks to Neural Machine Translations"

## Train SAA(transaformer) TAA(NMT with Luong attention) and MLM models

As HAA need to use pretrained models, we need to train the model first.

### train models
You can train the models by running the following command:
```
python SAA_train.python
python luong_attention.python
python bert_mlm_train.python
```
The dependencies of ``NLQF`` are:
```
transformers 4.9
nltk 3.4.5
numpy 1.19.5
pytorch 1.7.1
datasets 1.3
tensorflow 2.5
```
### Make attacks
After train the models mentioned above, we can run the following codes to generate adversarial examples
```
python test.py
```

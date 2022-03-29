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
transformers
nltk
numpy
pytorch
datasets
tensorflow
```
### Make attacks
After train the models mentioned above, we can run the following codes to generate adversarial examples
```
python test.py
```

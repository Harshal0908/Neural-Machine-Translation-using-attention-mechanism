"""https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import pandas as pd
import spacy,random

##Data Preparation and Pre-processing

spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")

def tokenize_german(text):
    return [token.text for token in spacy_german.tokenizer(text)]

def tokenize_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]

german = Field(tokenize = tokenize_german,lower=True, init_token = "<sos>", eos_token = "<eos>")

english = Field(tokenize = tokenize_english,lower=True, init_token="<sos>", eos_token = "<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts = (".de", ".en"),
                                                    fields=(german, english))

german.build_vocab(train_data,max_size=100000,min_freq = 2)
english.build_vocab(train_data,max_size=100000,min_freq = 2)

print(f"Unique tokens in source(de) vocabulary:{len(german.vocab)}")
print(f"Unique tokens in souecs(en) vocabulary:{len(english.vocab)}")


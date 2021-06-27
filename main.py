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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data,valid_data,test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.src),
                                                                      device = device)

#Building an Encoder

class Encoder(nn.Module):
    def __init__(self,input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
#YAHA SIRF INITIALISATION KIYA HAIN
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        #"embedding" is an object created for the Layer "Embedding"
        #so to perform embedding process we use its object
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
#YAHA PE ACTUALLY LSTM AUR EMBEDDING LAYER AUR DROPOUT LAYERS NE KAAM KIYA HAIN
    def forward(self, src):
        #PASSSED THE GERMAN LANGUAGE INTO EMBEDDING LAYER AND ADD DROPOUTS
        embedded = self.dropout(self.embedding(src))
        #OUTPUT OF THIS EMBEDDING LAYER IS PASSED ON TO THE LSTM LAYER
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden,cell


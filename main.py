"""https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350"""
""" https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb """

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

class Decoder(nn.Module):
    def __init__(self,output_dim,emb_dim,hid_dim,n_layers,dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)

    def forward(self,input,hidden,cell):

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        output,(hidden,cell) = self.rnn(embedded,(hidden,cell))

        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction,hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoders and Decoders must have equal Number of layers!"

    def forword(self,src,trg,teacher_forcing_ratio = 0.5):
        #src = [src_len,batch_size]
        #trg = [trg_len,batch_size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        #use english.vocab instead of self.decoder.output_dim
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len,batch_size,trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden,cell = self.encoder(src)

        #first input to the decoder is <sos> token
        input = trg[0,:]

        for t in range(1,trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input,hidden,cell)

            #place prediction in a tensor holding predictions for each token
            outputs[t] = output

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_forcing_ratio

            input = trg[t] if teacher_force else top1

        return outputs


#TRAINING THE SEQ2SEQ MODEL

INPUT_DIM = len(german.vocab)
OUTPUT_DIM = len(english.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM,ENC_EMB_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM,DEC_EMB_DIM,HID_DIM,N_LAYERS,DEC_DROPOUT)

model = Seq2Seq(enc,dec,device).to(device)

#INITIALIZING WEIGHTS IN PYTORCH from a uniform distribution between -0.08 and +0.08, i.e.(-0.08, 0.08).
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data,-0.08,0.08)
model.apply(init_weights) 
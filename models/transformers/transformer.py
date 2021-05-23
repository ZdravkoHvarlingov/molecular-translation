import torch
import torch.nn as nn
import math

from torch.nn.modules import linear
from common.vocabulary import Vocabulary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, encoder, ntoken, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # encoder
        self.encoder = encoder
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # decoder
        decoder_layers = nn.TransformerDecoderLayer(ntoken, nhead, nhid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, embed_size)


        self.embed_size = embed_size
        self.ntoken = ntoken

        self.linear = nn.Linear(embed_size, ntoken)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1

        # We already have the pretrained model, so no need to initialize weights!
        # self.encoder.weight.data.uniform_(-initrange, initrange)

        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, captions):
        embedded_caption = self.embedding(captions) 
        embedded_caption = self.linear(embedded_caption) 

        src = self.encoder(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        encoded_image = self.transformer_encoder(src)     
        encoded_image = self.linear(encoded_image)        


        outputs = self.transformer_decoder(torch.swapaxes(embedded_caption, 0,1), torch.swapaxes(encoded_image, 0,1))

        # [batch_size, seq_length, ntoken]
        outputs = torch.swapaxes(outputs,0,1)[:,1:,:]

        return outputs

    def generate_caption_from_predictions(self, predictions, vocab: Vocabulary):
        predicted_word_idx = predictions.argmax(dim=2)

        return predicted_word_idx


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
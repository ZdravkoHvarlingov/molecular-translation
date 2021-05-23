import torch
import torch.nn as nn
import math
from common.vocabulary import Vocabulary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, encoder, ntoken, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = encoder
        self.embed_size = embed_size
        self.ntoken = ntoken
        self.decoder = nn.Linear(embed_size, ntoken)
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

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, captions):
        # print(src_mask)
        # print("the shape of the image is: ", src.shape)
        src = self.encoder(src) * math.sqrt(self.embed_size)
        # print("This is the shape of the encoded image: ",src.shape)
        src = self.pos_encoder(src)
        # print("The encoded image has been encoded with positional encoding!: ",src.shape)
        
        output = self.transformer_encoder(src)

        seq_length = len(captions[0])-1
        batch_size = captions.size(0)

        outputs = torch.zeros(batch_size, seq_length, self.ntoken).to(device)
        
        for s in range(seq_length):
            
            
            decoded_output = self.decoder(self.dropout(output))


            outputs[:, s] = decoded_output[:,0,:]


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
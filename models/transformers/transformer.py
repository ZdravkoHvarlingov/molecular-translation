from matplotlib.pyplot import xcorr
from models.baseline.encoder_decoder import EncoderDecoder
import torch
import torch.nn as nn
import math

from common.vocabulary import Vocabulary
import numpy as np
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, ntoken, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # resnet encoder:
        self.linear = nn.Linear(2048, embed_size)

        # encoder
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # decoder
        decoder_layers = nn.TransformerDecoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, embed_size)


        self.embed_size = embed_size
        self.ntoken = ntoken
        self.predictor = nn.Linear(embed_size, ntoken)


        self.linear_2 = nn.Linear(12800, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, captions):
        
        # captions [4, x] batch_size, seq_length
        embedded_caption = self.embedding(captions) * math.sqrt(self.embed_size) # [4, x, 200] batch_size, seq_length, embedding dim
        embedded_caption = self.pos_encoder(embedded_caption) # [4, x, 200] batch_size, seq_length, embedding dim
        embedded_caption = embedded_caption.transpose(0,1) # [x, 4, 200] seq_length, batch_size, embedding dim

        # src [4, 1, 256, 256] batch_size, channels, height, width
        src = self.linear(src) #  [4, 1, 200] batch_size, resnet encoding
        src = self.pos_encoder(src) #  [4, 1, 200] batch_size, resnet encoding
        encoded_image = self.transformer_encoder(src) # [1, 4, 200] seq_length_image, batch_size, resnet encoding (down-scaled)

        tgt_mask = self.generate_square_subsequent_mask(embedded_caption.size(0)).to(device).transpose(0,1) # [x , x] seq_length, seq_length
        outputs = self.transformer_decoder(tgt = embedded_caption, tgt_mask = tgt_mask, memory = encoded_image) # [x, 4, 200]  seq_length, batch_size, embedding dim
        preds = self.predictor(outputs).permute(1, 2, 0) # [4, 42, x]  batch_size, ntoken, seq_length

        return preds


    def generate_caption(self, encoded_images, vocab=None, max_length = 300):
        encoded_images = self.linear(encoded_images) # [batch_size, 1, emb_dim]
        encoded_images = self.pos_encoder(encoded_images) # [batch_size, 1, emb_dim]
        encoded_images = self.transformer_encoder(encoded_images) # [batch_size, 1, emb_dim]

        batch_size = encoded_images.size(0)

        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        all_words = [word for i in range(batch_size)]
        words = torch.stack(all_words, dim=0).squeeze(dim=1)

        word_embeddings = self.embedding(words) # [batch_size, 1, emb_dim]

        captions = torch.zeros(batch_size, max_length).long().to(device) # [batch_size, max_length]

        encoded_images = encoded_images.transpose(0,1) # [1, batch_size, emb_dim]
        word_embeddings = word_embeddings.transpose(0,1) # [1, batch_size, emb_dim]

        
        for i in range(1,max_length):
            tgt_mask = self.generate_square_subsequent_mask(i).to(device)


            decoder_output = self.transformer_decoder(tgt=word_embeddings[:i, :],
                                        tgt_mask=tgt_mask,
                                        memory=encoded_images,
                                        ) # [i, 4, 200] i, batch_size, embedding_dim

            pred_proba_t = self.predictor(decoder_output)[-1, :] # [i, 4, 42] i, batch_size, ntoken -> the slice makes it [4, 42] batch_size, ntoken

            output_t = torch.topk(pred_proba_t, 1)[1].squeeze() # [4] batch_size (most likely tokens for each of the images in the batch)

            captions[:, i] = output_t # [4, t] batch_size, i

            word_embeddings = self.embedding(captions[:, :(i+1)]).transpose(0,1) # [i, batch_size, emb_dim]
        
        return captions


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
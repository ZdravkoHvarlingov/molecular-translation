import torch.nn as nn
from torch.nn.modules import transformer
from models.baseline.encoder_cnn import EncoderCNN
from models.baseline.decoder_rnn import DecoderRNN
from models.transformers.transformer import TransformerModel

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN(linear_dim = embed_size, transformer=False)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, ntoken, embed_size, nhead, nhid, nlayers, dropout=0.3):
        super().__init__()
        self.encoder = EncoderCNN(linear_dim = embed_size, transformer=True)
        self.decoder = TransformerModel(
            ntoken = ntoken, 
            embed_size = embed_size, 
            nhead = nhead, 
            nhid = nhid, 
            nlayers = nlayers, 
            dropout = dropout
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs

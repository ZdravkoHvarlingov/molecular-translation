import torch.nn as nn
from torch.nn.modules import transformer
from models.transformers.encoder_cnn import EncoderCNN
from models.baseline.decoder_rnn import DecoderRNN
from models.transformers.transformer import TransformerModel


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, ntoken, embed_size, nhead, nhid, nlayers, dropout=0.3):
        super().__init__()
        self.encoder = EncoderCNN(linear_dim = embed_size)
        self.decoder = TransformerModel(
            encoder = self.encoder,
            ntoken = ntoken, 
            embed_size = embed_size, 
            nhead = nhead, 
            nhid = nhid, 
            nlayers = nlayers, 
            dropout = dropout
        )
        
    def forward(self, images, captions):
        outputs = self.decoder(images, captions)
        
        return outputs

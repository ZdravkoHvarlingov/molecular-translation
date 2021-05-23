import torch.nn as nn
from models.EfficientNet.efficientnet_encoder import EfficientNetEncoder
from models.baseline.decoder_rnn import DecoderRNN


class EfficientEncoderDecoder(nn.Module):
    def __init__(self, model_name, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EfficientNetEncoder(model_name)
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

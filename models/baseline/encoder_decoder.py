import torch.nn as nn
from common.vocabulary import Vocabulary
from models.baseline.decoder_rnn import DecoderRNN
from models.baseline.encoder_cnn import EncoderCNN


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, sequence_length, vocab: Vocabulary, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            sequence_length=sequence_length,
            drop_prob=drop_prob,
            vocab=vocab
        )
        
    def forward(self, images, mode='train', captions=None):
        features = self.encoder(images)
        outputs = self.decoder(features, mode, captions)

        return outputs
        
    def generate_id_sequence_from_predictions(self, predictions):
        predicted_word_idx = predictions.argmax(dim=2)

        return predicted_word_idx

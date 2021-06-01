import torch.nn as nn
from common.vocabulary import Vocabulary
from models.transformers.encoder_cnn import EncoderCNN
from models.transformers.transformer import TransformerModel


class EncoderDecoderTransformer(nn.Module):

    NUM_HEADS = 2
    NUM_LAYERS = 6
    DROPOUT = 0.1
    EMBED_SIZE = 512

    def __init__(self, vocab: Vocabulary, sequence_length):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = TransformerModel(
            image_feature_size=self.encoder.IMAGE_FEATURE_SIZE, 
            embed_size=self.EMBED_SIZE,
            num_heads=self.NUM_HEADS,
            num_layers=self.NUM_LAYERS,
            vocab=vocab,
            sequence_length=sequence_length,
            dropout=self.DROPOUT)
        
    def forward(self, images, mode='train', captions=None):
        encoded_images = self.encoder(images)
        outputs = self.decoder(encoded_images, mode, captions)
        
        return outputs

    def generate_id_sequence_from_predictions(self, predictions):
        predicted_word_idx = predictions.argmax(dim=2)

        return predicted_word_idx

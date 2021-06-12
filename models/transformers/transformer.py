import math
import torch
import torch.nn as nn
from common.vocabulary import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):

    def __init__(self, image_feature_size, embed_size, num_heads, num_layers, vocab: Vocabulary, sequence_length, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.embed_size = embed_size
        self.embed_size_sqrt = math.sqrt(embed_size)

        self.cnn_to_embedding = nn.Linear(image_feature_size, embed_size)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.pred_linear = nn.Linear(embed_size, self.vocab_size)

        self.sequence_length = sequence_length
        self.sos_id = vocab.stoi['<SOS>']
        self.tgt_mask = self.generate_square_subsequent_mask(sequence_length).to(device)

        self.init_weights()

    def init_weights(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mode='train', captions=None):
        if mode == 'train' and captions is not None:
            return self._forward_train(src, captions)

        return self._forward_eval(src)
        
    def _forward_train(self, encoded_images, target_captions):
        encoded_images = self.cnn_to_embedding(encoded_images)
        encoded_images = self.pos_encoder(encoded_images)
        encoded_images = encoded_images.transpose(0, 1)
        #encoded_images = self.transformer_encoder(encoded_images)

        embedded_target = self.embedding(target_captions) * self.embed_size_sqrt
        embedded_target = self.pos_encoder(embedded_target)
        embedded_target = embedded_target.transpose(0,1)

        output = self.transformer_decoder(
            tgt = embedded_target, 
            memory = encoded_images, 
            tgt_mask = self.tgt_mask, # to avoid looking at the future tokens (the ones on the right)
        )
        
        preds = self.pred_linear(output).transpose(0, 1)
        # Right shifting is required so the transformer learns NOT to just copy the target input
        return preds[:,:-1,:]

    def _forward_eval(self, encoded_images):
        encoded_images = self.cnn_to_embedding(encoded_images)
        encoded_images = self.pos_encoder(encoded_images)
        encoded_images = encoded_images.transpose(0, 1)
        #encoded_images = self.transformer_encoder(encoded_images)
        
        batch_size = encoded_images.size(1)
        input_matrix = torch.zeros([batch_size, self.sequence_length], dtype=torch.long).to(device)
        input_matrix[:, 0] = self.sos_id

        seq_length = self.sequence_length - 1 # We do not want to predict the starting <SOS> token
        final_logits = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        for i in range(1, self.sequence_length):
            tgt_mask = self.tgt_mask[:i, :i]
            
            tgt = input_matrix[:, :i]
            embedded_tgt = self.embedding(tgt) * self.embed_size_sqrt
            embedded_tgt = self.pos_encoder(embedded_tgt)
            embedded_tgt = embedded_tgt.transpose(0,1)

            output = self.transformer_decoder(
                tgt=embedded_tgt, 
                memory=encoded_images, 
                tgt_mask=tgt_mask)
            
            preds = self.pred_linear(output).transpose(0, 1)
            preds = preds[:,-1,:] # the last timestep
            final_logits[:,i - 1,:] = preds

            predicted_word_idx = preds.argmax(dim=-1)
            input_matrix[:, i] = predicted_word_idx

        return final_logits


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

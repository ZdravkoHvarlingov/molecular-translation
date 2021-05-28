from common.vocabulary import Vocabulary
import torch
import torch.nn as nn
from models.baseline.attention import Attention


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, sequence_length, vocab: Vocabulary, drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.sos_id = vocab.stoi['<SOS>']
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, mode='train', captions=None):
        if mode == 'train' and captions is not None:
            return self._forward_train(features, captions)

        return self._forward_eval(features)

    def _forward_train(self, features, captions):
         # features shape = [batch_size, 64, 2048]
         # captions shape = [batch_size, max_length]
         # embeds shape = [batch_size, max_length, embed_size]
        embeds = self.embedding(captions)
        
        #initialize LSTM state
        h, c = self.init_hidden_state(features) #(batch_size, decoder_dim)
        
        # get the seq length to iterate, we remove 1 because we do NOT want to predict the <SOS> token
        seq_length = self.sequence_length - 1
        batch_size = captions.size(0)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        for s in range(seq_length):
            _, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h,c = self.lstm_cell(lstm_input, (h,c))
            
            output = self.fcn(self.dropout(h))
            preds[:, s] = output

        return preds

    def _forward_eval(self, features):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
 
        word = torch.tensor(self.sos_id).view(1,-1).to(device)
        all_words = [word] * batch_size
        words = torch.stack(all_words, dim=0).squeeze(dim=1)

        seq_length = self.sequence_length - 1
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        embeds = self.embedding(words)
        for i in range(seq_length):
            _, context = self.attention(features, h)
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h,c = self.lstm_cell(lstm_input, (h,c))
            
            output = self.fcn(self.dropout(h))

            output = output.view(batch_size,-1)
            
            #select the word
            
            # add the selected words to the predicted words tensor
            preds[:,i] = output
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0)).transpose(0,1)

        return preds

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        return h, c

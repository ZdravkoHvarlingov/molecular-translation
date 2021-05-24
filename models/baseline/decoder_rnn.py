from common.vocabulary import Vocabulary
import torch
import torch.nn as nn
from models.baseline.attention import Attention


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, captions):
         # features shape = [batch_size, 64, 2048]
         # captions shape = [batch_size, max_length]
         # embeds shape = [batch_size, max_length, embed_size]
        embeds = self.embedding(captions)
        
        #initialize LSTM state
        h, c = self.init_hidden_state(features) #(batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)
        
        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h,c = self.lstm_cell(lstm_input, (h,c))
            
            output = self.fcn(self.dropout(h))
            

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas
    
    def generate_caption_from_predictions(self, predictions, vocab: Vocabulary):
        predicted_word_idx = predictions.argmax(dim=2)

        return predicted_word_idx

    def generate_caption(self, features, max_length=1200, vocab=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        
        alphas=[]
        
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)

        #??
        captions=[]
        
        for i in range(max_length):
            alpha, context = self.attention(features, h)
            
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h,c = self.lstm_cell(lstm_input, (h,c))
            output = self.fcn(self.dropout(h))
            output = output.view(batch_size,-1)
            
            #select the word
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            if predicted_word_idx.item() == vocab.stoi["<EOS>"]:
                break
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
            
        return [vocab.itos[idx] for idx in captions], alphas
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        return h, c

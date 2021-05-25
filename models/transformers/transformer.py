import torch
import torch.nn as nn
import math

from torch.nn.modules import linear
from common.vocabulary import Vocabulary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, encoder, ntoken, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # resnet encoder:
        self.encoder = encoder
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


        self.linear_2 = nn.Linear(embed_size, ntoken)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        print("second line")
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1

        # We already have the pretrained model, so no need to initialize weights!
        # self.encoder.weight.data.uniform_(-initrange, initrange)

        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, captions, mode):
        # print("captions: ",captions.shape)
        embedded_caption = self.embedding(captions)  * math.sqrt(self.embed_size)
        # print("embedded_caption: ",embedded_caption.shape)
        embedded_caption = self.pos_encoder(embedded_caption)
        # print("embedded_caption: ",embedded_caption.shape)
        # embedded_caption = self.linear_2(embedded_caption) 
        # print("embedded_caption: ",embedded_caption.shape)


        # print("src: ",src.shape)
        # print("src: ",src.squeeze().reshape(src.size(0),-1).shape)
        src = self.encoder(src)
        # print("src:", src.shape)
        src = self.linear(src)
        # print("src: ",src.shape)
        # src = self.pos_encoder(src)
        # print("src: ",src.shape)
        encoded_image = self.transformer_encoder(src)    
        # print("encoded_image: ",encoded_image.shape) 
        # encoded_image = self.linear(encoded_image)        

        
        embedded_caption = embedded_caption.transpose(0,1)
        encoded_image = encoded_image.transpose(0,1)

        if mode == 'train':
            # print("The phase is train!!!!!!!!!!!!!!!!")
            # tgt_mask = self.generate_square_subsequent_mask(embedded_caption.size(0)).to(device).transpose(0,1)
            tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(
                embedded_caption.size(0)).to(device).transpose(0,1)
            # print("Embedded caption: ", embedded_caption.shape)
            # print("tgt_mask tgt_mask: ", tgt_mask.shape)
            # print("encoded_image : ", encoded_image.shape)

            outputs = self.transformer_decoder(tgt = embedded_caption, tgt_mask = tgt_mask, memory = encoded_image)
            # print(outputs.shape)
            # [batch_size, seq_length, ntoken]
            preds = self.predictor(outputs).permute(1, 2, 0)
            # print(self.predictor(outputs).shape)
            # print(self.predictor(outputs).permute(1, 2, 0).shape)

            # outputs = outputs.transpose(0,1)[:,1:,:]
            # print(outputs.shape)
            # softmax_outputs = self.softmax(outputs)
            
            return preds

        elif mode == 'eval':


        # d_model = 768
            bs = embedded_caption.size(1)
            sos_idx = 1
        # vocab_size = 10000
        # input_len = encoded_image.size(0)
            output_len = embedded_caption.size(0)

        # Define the model
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4).to(device)
        # encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
        #                                 num_layers=6).to(device)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4).to(device)
        # decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
        #                                 num_layers=6).to(device)

        # decoder_emb = nn.Embedding(vocab_size, d_model)
        # predictor = nn.Linear(d_model, vocab_size)

        # for a single batch x
            encoder_output = encoded_image.transpose(0,1)  # (bs, input_len, d_model)
            
            # initialized the input of the decoder with sos_idx (start of sentence token idx)
            output = torch.ones(bs, output_len).long().to(device)*sos_idx
            # print("The output of the encoder: ",encoder_output.shape)
            # print("Input to the decoder: ",output.shape)
            # print("The output length is: ", output_len)
            for t in range(1, output_len):
                # print(t)
                tgt_emb = self.embedding(output[:, :t]).transpose(0, 1)
                # print("tgt_emb: ", tgt_emb.shape)

                # tgt_mask_2 =  self.generate_square_subsequent_mask(2).to(device).transpose(0,1)
                tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(t).to(device).transpose(0,1)
                # print("tgt_mask_2: ", tgt_mask_2.shape)
                # tgt_mask =  self.generate_square_subsequent_mask(t).to(device).transpose(0,1)
                # print("tgt_mask: ", tgt_mask.shape)
                decoder_output = self.transformer_decoder(tgt=tgt_emb,
                                        memory=encoder_output,
                                        tgt_mask=tgt_mask)
                # print("decoder_output: ", decoder_output.shape)
                pred_proba_t = decoder_output[-1, :, :]
                # print("pred_proba_t: ", pred_proba_t.shape)
                output_t = pred_proba_t.data.topk(1)[1].squeeze()
                # print("output_t: ", pred_proba_t.data.topk(1))
                output[:, t] = output_t
                # print(output)
            #output (bs, output_len)
            return output

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
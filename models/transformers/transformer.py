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


        self.linear_2 = nn.Linear(12800, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1

        # We already have the pretrained model, so no need to initialize weights!
        # self.encoder.weight.data.uniform_(-initrange, initrange)

        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, captions, mode):
        
        # captions [4, x] batch_size, seq_length

        embedded_caption = self.embedding(captions)  * math.sqrt(self.embed_size) # [4, x, 200] batch_size, seq_length, embedding dim
        embedded_caption = self.pos_encoder(embedded_caption) # [4, x, 200] batch_size, seq_length, embedding dim


        # src [4, 1, 256, 256] batch_size, channels, height, width
        src = self.encoder(src) #  [4, 64, 2048] batch_size, resnet encoding
        src = self.linear(src) #  [4, 64, 200] batch_size, resnet encoding
        src = src.reshape(src.size(0),-1) # [4, 12800] batch_size, resnet encoding (flattened)
        src = self.linear_2(src) # [4, 200] batch_size, resnet encoding (down-scaled)
        src = src.unsqueeze(dim=0) # [1, 4, 200] seq_length_image, batch_size, resnet encoding (down-scaled)
        encoded_image = self.transformer_encoder(src) # [1, 4, 200] seq_length_image, batch_size, resnet encoding (down-scaled)

        # src = src.transpose(0,1) #  [64, 4, 200] batch_size, resnet encoding
        # encoded_image = self.transformer_encoder(src) # [64, 4, 200] seq_length_image, batch_size, resnet encoding (down-scaled)


        
        embedded_caption = embedded_caption.transpose(0,1) # [x, 4, 200] seq_length, batch_size, embedding dim

        # SOS, A, B
        # [True, False, False] -> SOS -> A
        # [True, True, False] -> SOS, A -> B
        # [True, True, True] -> SOS, A, B 

        if mode == 'train':

            tgt_mask = self.generate_square_subsequent_mask(embedded_caption.size(0)).to(device).transpose(0,1) # [x , x] seq_length, seq_length
            # print("Captions: ", embedded_caption.shape)
            # print("Images: ", encoded_image.shape)
            outputs = self.transformer_decoder(tgt = embedded_caption, tgt_mask = tgt_mask, memory = encoded_image) # [x, 4, 200]  seq_length, batch_size, embedding dim
            # print("Training!!!!!!!!!!!!!!!!!!!")
            # print("Decoder output shape", outputs.shape)
            # print("Decoder output: ",outputs)

            # x 4 42
            preds = self.predictor(outputs).permute(1, 2, 0) # [4, 42, x]  batch_size, ntoken, seq_length

            return preds

        elif mode == 'eval':
            bs = encoded_image.size(1)
            sos_idx = 1
            eos_idx = 2
            output_len = embedded_caption.size(0)

            encoder_output = encoded_image.transpose(0,1)  # (bs, input_len, d_model)
            
            # initialized the input of the decoder with sos_idx (start of sentence token idx)
            output = torch.ones(bs, output_len).long().to(device) * sos_idx
            preds = torch.ones(bs, output_len, self.ntoken).long().to(device) * sos_idx
            
            print("Output!!!!!!!!!!!!!!!!!!!!!! ",output)
            for t in range(1, output_len):
 
                tgt_emb = self.embedding(output[:, :t]).transpose(0, 1) # [t, 4, 200] t, batch_size, embedding_dim

                tgt_mask = self.generate_square_subsequent_mask(t).to(device).transpose(0,1) # [t, t] t, t

                decoder_output = self.transformer_decoder(tgt=tgt_emb,
                                        memory=encoder_output,
                                        tgt_mask=tgt_mask) # [t, 4, 200] t, batch_size, embedding_dim

                print("Decoder output shape", decoder_output.shape)
                print("Decoder output: ",decoder_output)

                pred_proba_t = self.predictor(decoder_output)[-1, :, :] # [t, 4, 42] t, batch_size, ntoken -> the slice makes it [4, 42] batch_size, ntoken
                # print("print(self.predictor(decoder_output)) : ",self.predictor(decoder_output))
                # print("print(self.predictor(decoder_output)) shape: ",self.predictor(decoder_output).shape)
                
                # print("Pred_proba_t shape: ",pred_proba_t.shape)
                # print("Pred_proba_t: ",pred_proba_t)
                output_t = pred_proba_t.data.topk(1)[1].squeeze() # [4] batch_size 
                print(pred_proba_t.data.topk(1))
                # print("Output_t shape: ", output_t.shape)
                # print("Output_t: ",output_t)

                output[:, t] = output_t # [4, t] batch_size, t
                print("output_t : ",output[:, t])
                print("output_t : ",output[:, t].shape)
                preds[:,t] = pred_proba_t # [4, t, 42] batch_size, t, ntoken
                print("preds : ",preds[:,t])
                print("preds : ",preds[:,t].shape)

            #output (bs, output_len)
            #preds (bs, output_len, ntoken)
            return output, preds

        elif mode == 'inference':

            bs = embedded_caption.size(1)
            sos_idx = 1
            eos_idx = 2
            output_len = embedded_caption.size(0)

        # for a single batch x
            encoder_output = encoded_image.transpose(0,1)  # (bs, input_len, d_model)
            
            # initialized the input of the decoder with sos_idx (start of sentence token idx)
            output = torch.ones(bs, output_len).long().to(device)*sos_idx
            preds = torch.ones(bs, output_len, self.ntoken).long().to(device)*sos_idx
            
            # print("The output of the encoder: ",encoder_output.shape)
            # print("Input to the decoder: ",output.shape)
            # print("The output length is: ", output_len)
            print("Creating the predictions!")
            for t in range(1, output_len):
                # print(t)
                # print(preds[:, :t, :])
                # print(output[:, :t])

                tgt_emb = self.embedding(output[:, :t]).transpose(0, 1)
                # print("tgt_emb: ", tgt_emb.shape)

                # tgt_mask_2 =  self.generate_square_subsequent_mask(2).to(device).transpose(0,1)
                tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(t).to(device).transpose(0,1)
                # print("tgt_mask_2: ", tgt_mask_2.shape)
                # tgt_mask =  self.generate_square_subsequent_mask(t).to(device).transpose(0,1)
                # print("tgt_mask: ", tgt_mask.shape)
                # print("encoder_output: ", encoder_output.shape)
                decoder_output = self.transformer_decoder(tgt=tgt_emb,
                                        memory=encoder_output,
                                        tgt_mask=tgt_mask)
                # print("decoder_output: ", decoder_output.shape)
                pred_proba_t = self.predictor(decoder_output)[-1, :, :]
                # print("pred_proba_t: ", self.predictor(decoder_output).shape)
                # print("pred_proba_t: ", pred_proba_t.shape)
                output_t = pred_proba_t.data.topk(1)[1].squeeze()
                # print("output_t: ", output_t.shape)
                # print(output.shape)
                
                output[:, t] = output_t
                
                # print("output slice: ",output[:,t].shape)
                # print("preds slice: ",preds[:,t].shape)
                # preds[:, t, :] = [output_t, pred_proba_t[0,:]]
                preds[:,t] = pred_proba_t
                
                # print("Top 1: ", pred_proba_t.data.topk(1))
                # print("Top 1 [1]: ", pred_proba_t.data.topk(1)[1].shape)

                # print(output)
            #output (bs, output_len)
            return output, preds

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
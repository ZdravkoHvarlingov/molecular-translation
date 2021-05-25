from math import pi
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from common.dataset import retrieve_evaluate_dataloader, retrieve_train_dataloader
from common.vocabulary import Vocabulary
from models.transformers.encoder_decoder import EncoderDecoderTransformer
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderDecoderTrainer:

    def __init__(self, embed_size=200, attention_dim=300, encoder_dim=2048, decoder_dim=300, batch_size=4):
        self.embed_size = embed_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.batch_size = batch_size
        self.vocab = Vocabulary()
    
    def train(self, dataframe, num_epochs=10, load_state_file=None, plot_metrics=False):
        torch.cuda.empty_cache()
        saved_params = None
        trained_epochs = 0
        if load_state_file is not None:
            saved_params = torch.load(load_state_file)
            self.embed_size = saved_params['embed_size']
            self.attention_dim = saved_params['attention_dim']
            self.encoder_dim = saved_params['encoder_dim']
            self.decoder_dim = saved_params['decoder_dim']
            trained_epochs = saved_params['num_epochs']

        
        print("Using Transformer model!")
        model = EncoderDecoderTransformer(
            ntoken = len(self.vocab),
            embed_size=self.embed_size,
            nhead = 2, 
            nhid = 200, 
            nlayers = 2, 
            dropout = 0.3
        ).to(device)


        print(len(self.vocab))

        # ntokens = len(vocab.stoi) # the size of vocabulary
        # emsize = 200 # embedding dimension
        # nhead = 2 # the number of heads in the multiheadattention models
        # nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
        # nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        # dropout = 0.2 # the dropout value


        if saved_params is not None:
            model.load_state_dict(saved_params['state_dict'])
        model.train()

        self._perform_training(dataframe, model, num_epochs, trained_epochs, plot_metrics)

    def _perform_training(self, dataframe, model, num_epochs, trained_epochs=0, plot_metrics=False):
        train_df, validation_df, _ = self._split_train_val_test(dataframe)

        
        dataloader = retrieve_train_dataloader(train_df, self.vocab, batch_size=self.batch_size)

        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), lr = 3e-4)
        vocab_size = len(self.vocab)

        train_losses, val_losses = [], []
        train_levenshteins, val_levenshteins = [], []
        

        for epoch in tqdm(range(trained_epochs + 1, trained_epochs + num_epochs + 1), position=0, leave=True):
            print("\n Epoch: ", epoch)


            print("Training! ")
            for image, captions in tqdm(dataloader):
                image, captions = image.to(device), captions.to(device)


                optimizer.zero_grad()
                
                outputs = model(image, captions, mode = "train")
                # targets = captions[:, 1:]
                # outputs = outputs[:,:,1:]

                # outputs = [4, 42, x] -> [4x, 42]
                # captions = [4, x] ->    [4x]
                # print("Ouptuts shape: ", outputs.shape)
                # print("captions shape: ", captions.shape)

                loss = loss_func(outputs, captions)
                # loss = loss_func(outputs.reshape(-1, vocab_size), captions.reshape(-1))
                
                print("The loss in the training phase is: ", loss.item())
                loss.backward()
                optimizer.step()

            model.eval()
            print(f'Training set evaluation:')
            train_loss, train_levenshtein = self._evaluate_model(model, train_df, mode="eval")
            train_losses.append(train_loss)
            train_levenshteins.append(train_levenshtein)

            print(f'Validation set evaluation:')
            val_loss, val_levenshtein = self._evaluate_model(model, validation_df, mode="eval")
            val_losses.append(val_loss)
            val_levenshteins.append(val_levenshtein)
            model.train()
                
                
            if plot_metrics and train_losses and train_levenshteins and val_losses and val_levenshteins:

                import os
                os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
                self._plot_metrics(train_losses, train_levenshteins, val_losses, val_levenshteins)

            self._save_model(model, epoch)
        
    def _evaluate_model(self, model, dataframe, mode):
        vocab = self.vocab
        vocab_size = len(vocab)

        verbose = True
        losses = []
        levenshteins = []
        with torch.no_grad():
            loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
            dataloader = retrieve_evaluate_dataloader(dataframe, vocab, batch_size=self.batch_size)

            for image, captions in tqdm(dataloader, position=0, leave=True):
                image, captions = image.to(device), captions.to(device)
                
                outputs, preds = model(image, captions, mode=mode)

                targets = captions[:, 1:]
                outputs = outputs[:, 1:]
                # preds = preds[:,1:,:]
                # print("Targets shape: ",targets.shape)
                # print("outputs shape: ",outputs.shape)
                # print("Preds shape: ",preds.shape)
                # print("captions shape: ",captions.shape)


                loss = loss_func(preds.transpose(1,2).float(), captions)

                losses.append(loss.detach().item())
                outputs = outputs[:, 1:]
                predicted_word_idx_list = outputs
                
                # predicted_word_idx_list = model.decoder.generate_caption_from_predictions(outputs, vocab)
                batch_levenshtein = self._calc_batch_levenshtein(predicted_word_idx_list, targets, verbose)
                levenshteins.append(batch_levenshtein) 

                verbose = False            
            print("Mean loss: ", np.mean(losses))
            print("Mean levensthein: ", np.mean(levenshteins))
            return np.mean(losses), np.mean(levenshteins)

    def _calc_batch_levenshtein(self, predicted_word_idx, targets, verbose=False):
        predicted_word_idx = list(predicted_word_idx.cpu().numpy())
        targets = list(targets.cpu().numpy())

        distances = []
        for index, predicted_sentence_word_idx in enumerate(predicted_word_idx):
            sentence_to_str = self._word_idx_to_caption_sentence(predicted_sentence_word_idx)
            target_sentence = self._word_idx_to_caption_sentence(targets[index])
            levenshtein_metric = levenshtein_distance(target_sentence, sentence_to_str)
            distances.append(levenshtein_metric)
            
            if verbose:
                print(f'\nPredicted: {sentence_to_str}')
                print(f'Target: {target_sentence}')
                print(f'Levenshtein distance: {levenshtein_metric}\n')
                verbose = False

        return np.mean(distances)

    def _word_idx_to_caption_sentence(self, word_idx_list):
        vocab = self.vocab
        pad_idx = vocab.stoi['<PAD>']
        eos_idx = vocab.stoi['<EOS>']
        
        words = []
        for word_idx in word_idx_list:
            if word_idx == pad_idx:
                continue
            
            words.append(vocab.itos[word_idx])
            if word_idx == eos_idx:
                break

        return ''.join(words)

    def _split_train_val_test(self, dataframe):
        train=dataframe.sample(frac=0.7,random_state=11) #random state is a seed value
        val_test=dataframe.drop(train.index)
        val = val_test.sample(frac=0.5, random_state=12)
        test = val_test.drop(val.index)

        return train, val, test

    def _save_model(self, model, num_epochs):
        model_state = {
            'num_epochs':num_epochs,
            'embed_size':self.embed_size,
            'vocab_size':len(self.vocab),
            'attention_dim': self.attention_dim,
            'encoder_dim': self.encoder_dim,
            'decoder_dim': self.decoder_dim,
            'state_dict':model.state_dict()
        }

        
        filepath = Path(f'saved_models/')
        filepath.mkdir(parents=True, exist_ok=True)
        torch.save(model_state,f'saved_models/transformer_model_state_epoch_{num_epochs}.pth')

    def _plot_metrics(self, train_losses, train_levenshteins, val_losses, val_levenshteins):
        print("Plotting metrics!")
        
        plt.title("Train/Validation Loss")
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="validation")
        plt.legend()
        plt.savefig('TrainValidationLoss.png')
        plt.clf()
        
        plt.title("Train/Validation Levenshtein")
        plt.plot(train_levenshteins, label="train")
        plt.plot(val_levenshteins, label="validation")
        plt.legend()
        plt.savefig('TrainValidationLevenshtein.png')
        plt.clf()
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.dataset import (retrieve_evaluate_dataloader,
                            retrieve_train_dataloader)
from common.vocabulary import Vocabulary
from Levenshtein import distance as levenshtein_distance
from models.transformers.encoder_decoder_transformer import EncoderDecoderTransformer
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderDecoderTrainer:

    def __init__(self, sequence_length=405, batch_size=4):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = Vocabulary()
    
    def train(self, dataframe, num_epochs=10, load_state_file=None, plot_metrics=False):
        torch.cuda.empty_cache()
        saved_params = None
        trained_epochs = 0

        if load_state_file is not None:
            saved_params = torch.load(load_state_file)
            trained_epochs = saved_params['num_epochs']

        print("Using Transformer model!")
        model = EncoderDecoderTransformer(
            sequence_length=self.sequence_length,
            vocab = self.vocab
        ).to(device)

        if saved_params is not None:
            model.load_state_dict(saved_params['state_dict'])
        model.train()

        self._perform_training(dataframe, model, num_epochs, trained_epochs, plot_metrics)

    def _perform_training(self, dataframe, model, num_epochs, trained_epochs=0, plot_metrics=False):
        train_df, validation_df, _ = self._split_train_val_test(dataframe)
        dataloader = retrieve_train_dataloader(train_df, self.vocab, batch_size=self.batch_size, sequence_length=self.sequence_length)

        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), lr = 3e-4)
        vocab_size = len(self.vocab)

        train_losses, val_losses = [], []
        train_levenshteins, val_levenshteins = [], []
        for epoch in tqdm(range(trained_epochs + 1, trained_epochs + num_epochs + 1), position=0, leave=True):
            print("\n Epoch: ", epoch)


            print("Training! ")
            for images, captions in tqdm(dataloader):
                images, captions = images.to(device), captions.to(device)

                optimizer.zero_grad()

                targets = captions[:, 1:]
                preds = model(images, 'train', captions)

                loss = loss_func(preds.reshape(-1, vocab_size), targets.reshape(-1))
                
                loss.backward()
                optimizer.step()

            model.eval()
            print(f'Training set evaluation:')
            train_loss, train_levenshtein = self._evaluate_model(model, train_df)
            train_losses.append(train_loss)
            train_levenshteins.append(train_levenshtein)
            print(f'Training set avg loss: {train_loss}')
            print(f'Training set avg levenshteins: {train_levenshtein}')

            print(f'Validation set evaluation:')
            val_loss, val_levenshtein = self._evaluate_model(model, validation_df)
            val_losses.append(val_loss)
            val_levenshteins.append(val_levenshtein)
            print(f'Validation set avg loss: {val_loss}')
            print(f'Validation set avg levenshteins: {val_levenshtein}')
            
            model.train()
                
            if plot_metrics and train_losses and train_levenshteins and val_losses and val_levenshteins:
                self._plot_metrics(train_losses, train_levenshteins, val_losses, val_levenshteins)

            self._save_model(model, epoch)
        
    def _evaluate_model(self, model, dataframe):
        vocab = self.vocab
        vocab_size = len(vocab)

        verbose = True
        losses = []
        levenshteins = []
        with torch.no_grad():
            loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
            dataloader = retrieve_evaluate_dataloader(dataframe, vocab, batch_size=self.batch_size, sequence_length=self.sequence_length)

            for image, captions in tqdm(dataloader, position=0, leave=True):
                image, captions = image.to(device), captions.to(device)

                targets = captions[:, 1:]
                preds = model(image, 'eval')

                loss = loss_func(preds.reshape(-1, vocab_size), targets.reshape(-1))
                losses.append(loss.detach().item())
                
                predicted_word_idx_list = model.generate_id_sequence_from_predictions(preds)
                batch_levenshtein = self._calc_batch_levenshtein(predicted_word_idx_list, targets, verbose)
                levenshteins.append(batch_levenshtein) 

                verbose = False

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

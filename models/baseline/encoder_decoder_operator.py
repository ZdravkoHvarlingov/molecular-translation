import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from common.dataset import retrieve_train_dataloader
from common.training_utils import TrainingUtils
from common.vocabulary import Vocabulary
from models.baseline.encoder_decoder import EncoderDecoder
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderDecoderOperator:

    def __init__(self, embed_size=200, attention_dim=300, encoder_dim=2048, decoder_dim=300, batch_size=4, sequence_length=405):
        self.embed_size = embed_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = Vocabulary()
    
    def train(self, data_csv_path, num_epochs=10, load_state_file=None, plot_metrics=False):
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

        print("Using Bahdanau Attention model!")
        model = EncoderDecoder(
        embed_size=self.embed_size,
        vocab_size=len(self.vocab),
        attention_dim=self.attention_dim,
        encoder_dim=self.encoder_dim,
        decoder_dim=self.decoder_dim,
        sequence_length=self.sequence_length,
        vocab=self.vocab
        ).to(device)

        if saved_params is not None:
            model.load_state_dict(saved_params['state_dict'])
        model.train()

        self._perform_training(data_csv_path, model, num_epochs, trained_epochs, plot_metrics)

    def _perform_training(self, data_csv_path, model, num_epochs, trained_epochs=0, plot_metrics=False):
        train_df, validation_df, _ = TrainingUtils.split_train_val_test(data_csv_path)
        dataloader = retrieve_train_dataloader(
            train_df,
            self.vocab,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length)

        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), lr = 3e-4)
        vocab_size = len(self.vocab)

        train_losses, val_losses = [], []
        train_levenshteins, val_levenshteins = [], []

        for epoch in tqdm(range(trained_epochs + 1, trained_epochs + num_epochs + 1), position=0, leave=True):
            print("\n Epoch: ", epoch)
            for image, captions in tqdm(dataloader):
                image, captions = image.to(device), captions.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(image, 'train', captions)
                targets = captions[:, 1:] # Remove <SOS> token

                loss = loss_func(outputs.view(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                optimizer.step()

            model.eval()
            print(f'Training set evaluation:')
            train_loss, train_levenshtein = TrainingUtils.evaluate_model_on_dataset(
                model, train_df, self.sequence_length, self.batch_size, self.vocab, 'eval')
            train_losses.append(train_loss)
            train_levenshteins.append(train_levenshtein)

            print(f'Validation set evaluation:')
            val_loss, val_levenshtein = TrainingUtils.evaluate_model_on_dataset(
                model, validation_df, self.sequence_length, self.batch_size, self.vocab, 'eval')
            val_losses.append(val_loss)
            val_levenshteins.append(val_levenshtein)

            model.train()    
                
            if plot_metrics and train_losses and train_levenshteins and val_losses and val_levenshteins:
                TrainingUtils.plot_metrics(train_losses, train_levenshteins, val_losses, val_levenshteins)

            self._save_model(model, epoch)

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
        torch.save(model_state,f'saved_models/seq2seq_rnn_model_state_epoch_{num_epochs}.pth')

    def predict(self, data_csv_path: str, model_state_file):
        torch.cuda.empty_cache()
        saved_params = torch.load(model_state_file)
        self.embed_size = saved_params['embed_size']
        self.attention_dim = saved_params['attention_dim']
        self.encoder_dim = saved_params['encoder_dim']
        self.decoder_dim = saved_params['decoder_dim']

        model = EncoderDecoder(
            embed_size=self.embed_size,
            vocab_size=len(self.vocab),
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            sequence_length=self.sequence_length,
            vocab=self.vocab
        ).to(device)
        model.load_state_dict(saved_params['state_dict'])
        model.eval()

        dataframe = pd.read_csv(data_csv_path)
        results_file_name = data_csv_path.replace(".csv", "_results.csv")
        TrainingUtils.predict_on_dataset(model, dataframe, self.batch_size, self.vocab, results_file_name)
    
    def evaluate(self, data_csv_path: str, model_state_file):
        torch.cuda.empty_cache()
        saved_params = torch.load(model_state_file)
        self.embed_size = saved_params['embed_size']
        self.attention_dim = saved_params['attention_dim']
        self.encoder_dim = saved_params['encoder_dim']
        self.decoder_dim = saved_params['decoder_dim']

        model = EncoderDecoder(
            embed_size=self.embed_size,
            vocab_size=len(self.vocab),
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            sequence_length=self.sequence_length,
            vocab=self.vocab
        ).to(device)
        model.load_state_dict(saved_params['state_dict'])
        model.eval()

        dataframe = pd.read_csv(data_csv_path)
        TrainingUtils.evaluate_model_levenshtein(model, dataframe, self.sequence_length, self.batch_size, self.vocab)

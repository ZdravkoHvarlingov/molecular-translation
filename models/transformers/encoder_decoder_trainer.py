import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from common.dataset import retrieve_train_dataloader
from common.training_utils import TrainingUtils
from common.vocabulary import Vocabulary
from models.transformers.encoder_decoder_transformer import EncoderDecoderTransformer
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderDecoderTrainer:

    LEARNING_RATE = 1e-4

    def __init__(self, sequence_length=405, batch_size=4):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = Vocabulary()
        self.vocab_size = len(self.vocab)
    
    def train(self, data_csv_path, num_epochs=10, load_state_file=None, plot_metrics=False):
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

        self._perform_training(data_csv_path, model, num_epochs, trained_epochs, plot_metrics)

    def _perform_training(self, data_csv_path, model, num_epochs, trained_epochs=0, plot_metrics=False):
        train_df, validation_df, _ = TrainingUtils.split_train_val_test(data_csv_path)
        dataloader = retrieve_train_dataloader(
            train_df,
            self.vocab,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length
        )

        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), self.LEARNING_RATE)

        train_losses, val_losses = [], []
        train_levenshteins, val_levenshteins = [], []
        for epoch in tqdm(range(trained_epochs + 1, trained_epochs + num_epochs + 1), position=0, leave=True):
            print("\nEpoch: ", epoch)

            for images, captions in tqdm(dataloader):
                images, captions = images.to(device), captions.to(device)

                optimizer.zero_grad()

                targets = captions[:, 1:]
                preds = model(images, 'train', captions)
                loss = loss_func(preds.reshape(-1, self.vocab_size), targets.reshape(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            model.eval()
            print(f'Training set evaluation:')
            train_loss, train_levenshtein = TrainingUtils.evaluate_model_on_dataset(
                model, train_df, self.sequence_length, self.batch_size, self.vocab, 'train'
            )
            train_losses.append(train_loss)
            train_levenshteins.append(train_levenshtein)

            print(f'Validation set evaluation:')
            val_loss, val_levenshtein = TrainingUtils.evaluate_model_on_dataset(
                model, validation_df, self.sequence_length, self.batch_size, self.vocab, 'train')
            val_losses.append(val_loss)
            val_levenshteins.append(val_levenshtein)
            model.train()
                
            if plot_metrics and train_losses and train_levenshteins and val_losses and val_levenshteins:
                TrainingUtils.plot_metrics(train_losses, train_levenshteins, val_losses, val_levenshteins)
        
            if epoch % 1 == 0:
                model.eval()
                print(f'Training set inference evaluation:')
                train_loss, train_levenshtein = TrainingUtils.evaluate_model_on_dataset(
                    model, train_df, self.sequence_length, self.batch_size, self.vocab, 'eval'
                )

                print(f'Validation set inference evaluation:')
                val_loss, val_levenshtein = TrainingUtils.evaluate_model_on_dataset(
                    model, validation_df, self.sequence_length, self.batch_size, self.vocab, 'eval')
                model.train()

            self._save_model(model, epoch)

    def _save_model(self, model, num_epochs):
        model_state = {
            'num_epochs':num_epochs,
            'state_dict':model.state_dict()
        }
        
        filepath = Path(f'saved_models/')
        filepath.mkdir(parents=True, exist_ok=True)
        torch.save(model_state,f'saved_models/transformer_model_state.pth')

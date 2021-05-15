import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from common.dataset import retrieve_evaluate_dataloader, retrieve_train_dataloader
from common.vocabulary import Vocabulary
from models.baseline.encoder_decoder import EncoderDecoder
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderDecoderTrainer:

    def __init__(self, embed_size=200, attention_dim=300, encoder_dim=2048, decoder_dim=300, batch_size=8):
        self.embed_size = embed_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.batch_size = batch_size
        self.vocab = Vocabulary()
    
    def train(self, dataframe, num_epochs=10, load_state_file=None):
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

        model = EncoderDecoder(
            embed_size=self.embed_size,
            vocab_size=len(self.vocab),
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim
        ).to(device)

        if saved_params is not None:
            model.load_state_dict(saved_params['state_dict'])
        model.train()

        self._perform_training(dataframe, model, num_epochs, trained_epochs)

    def _perform_training(self, dataframe, model, num_epochs, trained_epochs=0):
        train_df, validation_df, test_df = self._split_train_val_test(dataframe)
        dataloader = retrieve_train_dataloader(train_df, self.vocab, batch_size=4)

        print_every = 10
        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), lr = 3e-4)
        vocab_size = len(self.vocab)

        iteration = 0
        train_losses = []
        val_losses = []
        for epoch in tqdm(range(trained_epochs + 1, trained_epochs + num_epochs + 1)):
            for image, captions in tqdm(dataloader, position=0, leave=True):
                #imageTensor, captions = imageTensor.to(device), captions.to(device)
                image, captions = image.to(device), captions.to(device)

                optimizer.zero_grad()
                
                outputs, _ = model(image, captions)
                targets = captions[:, 1:]
                
                loss = loss_func(outputs.view(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                
                if (iteration + 1) % print_every == 0:
                    model.eval()
                    train_loss = self._evaluate_model(self.vocab, model, train_df)
                    train_losses.append(train_loss)
                    val_loss = self._evaluate_model(self.vocab, model, validation_df)
                    val_losses.append(val_loss)
                    model.train()

                    print(f'Train loss: {train_loss}')
                    print(f'Validation loss: {val_loss}')
                
                iteration += 1
                    
            self.save_model(model, epoch)
        
    def _evaluate_model(self, vocab: Vocabulary, model, dataframe):
        vocab_size = len(vocab)
        losses = []
        with torch.no_grad():
            loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
            dataloader = retrieve_evaluate_dataloader(dataframe, vocab, batch_size=4, shuffle=False)
            
            for image, captions in tqdm(dataloader, position=0, leave=True):
                #imageTensor, captions = imageTensor.to(device), captions.to(device)
                image, captions = image.to(device), captions.to(device)
                outputs, _ = model(image, captions)
                targets = captions[:, 1:]
                
                loss = loss_func(outputs.view(-1, vocab_size), targets.reshape(-1))
                losses.append(loss.detach().item())

            dataiter = iter(dataloader)
            img, original_captions  = next(dataiter)
            features = model.encoder(img[0:1].to(device))
            caps, _ = model.decoder.generate_caption(features, vocab=self.vocab)
            caption = ''.join(caps)
            
            original_caption = [self.vocab.itos[token_id] for token_id in original_captions[0].numpy()[1:] if self.vocab.itos[token_id] != '<PAD>']
            original_caption = ''.join(original_caption)
            print(f'Original caption: {original_caption}')
            print(f'Generated caption: {caption}')

            levenshtein_metric = levenshtein_distance(original_caption, caption)
            print(f'Levenshtein distance: {levenshtein_metric}')

            return np.mean(losses)

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

        torch.save(model_state,f'saved_models/attention_model_state_epoch_{num_epochs}.pth')

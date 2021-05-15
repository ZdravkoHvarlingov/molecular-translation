import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from common.dataset import retrive_dataloader
from common.vocabulary import Vocabulary
from models.baseline.encoder_decoder import EncoderDecoder
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
    
    def train(self, dataframe, num_epochs=10):
        torch.cuda.empty_cache()

        dataloader = retrive_dataloader(dataframe, self.vocab)
        model = EncoderDecoder(
            embed_size=self.embed_size,
            vocab_size=len(self.vocab),
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim
        ).to(device)

        self._perform_training(dataloader, model, num_epochs)


    def _perform_training(self, dataloader, model, num_epochs):
        print_every = 10
        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])
        optimizer = optim.Adam(model.parameters(), lr = 3e-4)
        vocab_size = len(self.vocab)

        iteration = 0
        for epoch in tqdm(range(1, num_epochs + 1)):
            for image, captions in tqdm(dataloader, position=0, leave=True):
                #imageTensor, captions = imageTensor.to(device), captions.to(device)
                image, captions = image.to(device), captions.to(device)

                optimizer.zero_grad()
                
                outputs, attentions = model(image, captions)
                targets = captions[:, 1:]
                
                loss = loss_func(outputs.view(-1, vocab_size), targets.reshape(-1))
                
                loss.backward()
                
                optimizer.step()
                
                if (iteration + 1) % print_every == 0:
                    print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
                    
                    model.eval()
                    
                    with torch.no_grad():
                        dataiter = iter(dataloader)
                        img, original_captions  = next(dataiter)
                        features = model.encoder(img[0:1].to(device))
                        caps, alphas = model.decoder.generate_caption(features, vocab=self.vocab)
                        caption = ''.join(caps)
                        
                        original_caption = [self.vocab.itos[token_id] for token_id in original_captions[0].numpy() if self.vocab.itos[token_id] != '<PAD>']
                        original_caption = ''.join(original_caption)
                        print(f'Original caption: {original_caption}')
                        print(f'Generated caption: {caption}')
                        plt.imshow(img[0].permute(1, 2, 0), cmap='gray')
                        
                    model.train()
                
                iteration += 1
                    
            self.save_model(model, epoch)


    def save_model(self, model, num_epochs):
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

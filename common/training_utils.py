import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

from common.dataset import retrieve_evaluate_dataloader
from common.vocabulary import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainingUtils:
    
    @staticmethod
    def split_train_val_test(dataframe):
        train=dataframe.sample(frac=0.7,random_state=11) #random state is a seed value
        val_test=dataframe.drop(train.index)
        val = val_test.sample(frac=0.5, random_state=12)
        test = val_test.drop(val.index)

        return train, val, test

    @staticmethod
    def calc_batch_levenshtein(predicted_word_idx, targets, vocab: Vocabulary, verbose=False):
        predicted_word_idx = list(predicted_word_idx.cpu().numpy())
        targets = list(targets.cpu().numpy())

        distances = []
        for index, predicted_sentence_word_idx in enumerate(predicted_word_idx):
            sentence_to_str = TrainingUtils.word_idx_to_caption_sentence(predicted_sentence_word_idx, vocab)
            target_sentence = TrainingUtils.word_idx_to_caption_sentence(targets[index], vocab)
            levenshtein_metric = levenshtein_distance(target_sentence, sentence_to_str)
            distances.append(levenshtein_metric)
            
            if verbose:
                print(f'\nPredicted: {sentence_to_str}')
                print(f'Target: {target_sentence}')
                print(f'Levenshtein distance: {levenshtein_metric}\n')
                verbose = False

        return np.mean(distances)

    @staticmethod
    def word_idx_to_caption_sentence(word_idx_list, vocab: Vocabulary):
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

    @staticmethod
    def plot_metrics(train_losses, train_levenshteins, val_losses, val_levenshteins):
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

    @staticmethod
    def evaluate_model_on_dataset(model, dataframe, sequence_length, batch_size, vocab, mode='train'):
        vocab_size = len(vocab)

        losses = []
        levenshteins = []
        with torch.no_grad():
            loss_func = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
            dataloader = retrieve_evaluate_dataloader(
                dataframe,
                vocab,
                batch_size=batch_size,
                sequence_length=sequence_length)
            
            verbose = True
            for image, captions in tqdm(dataloader, position=0, leave=True):
                image, captions = image.to(device), captions.to(device)

                targets = captions[:, 1:]
                if mode == 'train':
                    preds = model(image, 'train', captions)
                else:
                    preds = model(image, 'eval')

                loss = loss_func(preds.reshape(-1, vocab_size), targets.reshape(-1))
                losses.append(loss.detach().item())
                
                predicted_word_idx_list = model.generate_id_sequence_from_predictions(preds)
                batch_levenshtein = TrainingUtils.calc_batch_levenshtein(predicted_word_idx_list, targets, vocab, verbose)
                levenshteins.append(batch_levenshtein) 

                verbose = False

            average_loss = np.mean(losses)
            print(f'Dataset average loss: {average_loss}')
            average_levenshtein = np.mean(levenshteins)
            print(f'Dataset average levenshtein: {average_levenshtein}')

            return np.mean(losses), np.mean(levenshteins)

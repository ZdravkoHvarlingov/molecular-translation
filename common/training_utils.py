import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

from common.dataset import retrieve_evaluate_dataloader
from common.dataset_inference import retrieve_inference_dataloader
from common.vocabulary import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainingUtils:
    
    TRAIN_SIZE = 0.7

    @staticmethod
    def split_train_val_test(data_csv_path):
        train_file, val_file, test_file = TrainingUtils.create_split_file_names(data_csv_path)
        
        if os.path.isfile(train_file) and os.path.isfile(val_file) and os.path.isfile(test_file):
            print("No split required - train, validation and test files already present")
            return pd.read_csv(train_file), pd.read_csv(val_file), pd.read_csv(test_file)
        
        print("Split required - some or all of the train, validation and test files are missing")
        
        dataframe = pd.read_csv(data_csv_path)
        train=dataframe.sample(frac=TrainingUtils.TRAIN_SIZE,random_state=11) #random state is a seed value
        val_test=dataframe.drop(train.index)
        val = val_test.sample(frac=0.5, random_state=12)
        test = val_test.drop(val.index)

        train.to_csv(train_file)
        val.to_csv(val_file)
        test.to_csv(test_file)

        return train, val, test

    @staticmethod
    def create_split_file_names(train_csv_path: str):
        without_extension = train_csv_path.replace(".csv", "", 1)

        return f'{without_extension}_train.csv', f'{without_extension}_val.csv', f'{without_extension}_test.csv'

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
    def batch_idx_sequences_to_sentenses(predicted_word_idx_sequence, vocab: Vocabulary, verbose=False):
        predicted_word_idx = list(predicted_word_idx_sequence.cpu().numpy())

        sentences = []
        for predicted_sentence_word_idx in predicted_word_idx:
            sentence_to_str = TrainingUtils.word_idx_to_caption_sentence(predicted_sentence_word_idx, vocab)
            sentences.append(sentence_to_str)
            if verbose:
                print(f'Predicted: {sentence_to_str}')

        return sentences

    @staticmethod
    def word_idx_to_caption_sentence(word_idx_list, vocab: Vocabulary):
        pad_idx = vocab.stoi['<PAD>']
        eos_idx = vocab.stoi['<EOS>']
        
        words = []
        for word_idx in word_idx_list:
            if word_idx == pad_idx:
                continue

            if word_idx == eos_idx:
                break
            words.append(vocab.itos[word_idx])
            
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
    def evaluate_model_on_dataset(model, dataframe, sequence_length: int, batch_size: int, vocab, mode='train'):
        """
        Evaluates the specified model with the specified dataset.
        Returns the average loss and levenshtein distance of the dataset across all batches of the specified batch size.

        Parameters
        ----------
        model : torch.nn.Module
            A model which is intended to predict InChI formula based on an image.
        dataframe : pandas.Dataframe
            The dataset to be evaluated.
        sequence_length : int
            The maximum sequence length based on the InChI formulas in the dataset.
        batch_size : int
        vocab : Vocabulary
        mode : str
            Mode parameter specifies the forward method of the model.
            In case of 'train' the model will receive the captions as an input to the forward method which is much faster but usually applies teacher forcing.
            In case of 'eval' the model will NOT receive the captions as an input to the forward method which is much slower but shows the real accuracy of the model.
        """
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

    @staticmethod
    def evaluate_model_levenshtein(model, dataframe, sequence_length: int, batch_size: int, vocab):
        levenshteins = []
        with torch.no_grad():
            dataloader = retrieve_evaluate_dataloader(
                dataframe,
                vocab,
                batch_size=batch_size,
                sequence_length=sequence_length)
            
            for image, captions in tqdm(dataloader, position=0, leave=True):
                image, captions = image.to(device), captions.to(device)

                targets = captions[:, 1:]
                preds = model(image, 'eval')
                
                predicted_word_idx_list = model.generate_id_sequence_from_predictions(preds)
                batch_levenshtein = TrainingUtils.calc_batch_levenshtein(predicted_word_idx_list, targets, vocab, True)
                levenshteins.append(batch_levenshtein) 

            average_levenshtein = np.mean(levenshteins)
            print(f'Dataset average levenshtein: {average_levenshtein}')

            return np.mean(levenshteins)

    @staticmethod
    def predict_on_dataset(model, dataframe, batch_size: int, vocab, output_file):
        with torch.no_grad():
            dataloader = retrieve_inference_dataloader(
                dataframe,
                batch_size=batch_size)
            
            sentences = []
            for images in tqdm(dataloader, position=0, leave=True):
                images = images[0].to(device)

                preds = model(images, 'eval')

                predicted_word_idx_list = model.generate_id_sequence_from_predictions(preds)
                batch_sentences = TrainingUtils.batch_idx_sequences_to_sentenses(predicted_word_idx_list, vocab, True)
                sentences.extend(batch_sentences)

            with open(output_file, 'w') as result_file:
                result_file.writelines([f'"{sentence}"\n' for sentence in sentences])

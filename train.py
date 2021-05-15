import pandas as pd

from models.baseline.encoder_decoder_trainer import EncoderDecoderTrainer


def train():
    trainer = EncoderDecoderTrainer()
    dataframe = pd.read_csv("bms-molecular-translation/extended_data_1000.csv")
    trainer.train(dataframe, 1000, 'saved_models/attention_model_state_epoch_32.pth')

if __name__ == '__main__':
    train()

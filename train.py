import pandas as pd

from models.baseline.encoder_decoder_trainer import EncoderDecoderTrainer


def train():
    trainer = EncoderDecoderTrainer()
    dataframe = pd.read_csv("..\\bms-molecular-translation/extended_data_200.csv")
    trainer.train(dataframe, 1000, plot_metrics=True)

if __name__ == '__main__':
    train()

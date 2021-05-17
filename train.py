import pandas as pd
from pandas.core.frame import DataFrame

from models.baseline.encoder_decoder_trainer import EncoderDecoderTrainer


def train():
    trainer = EncoderDecoderTrainer()
    dataframe = pd.read_csv("..\\bms-molecular-translation\extended_data_200.csv")
    print(dataframe.head())


    trainer.train(dataframe, 1000, plot_metrics=True)
if __name__ == '__main__':
    train()

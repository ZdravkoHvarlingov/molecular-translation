import pandas as pd

from models.baseline.encoder_decoder_trainer import EncoderDecoderTrainer as BahdanauAttention
from models.transformers.encoder_decoder_trainer import EncoderDecoderTrainer as Transformer


def train():
    # trainer = BahdanauAttention(batch_size=4)
    trainer = Transformer(batch_size=4)
    dataframe = pd.read_csv("bms-molecular-translation/extended_data_30.csv")
    trainer.train(dataframe, num_epochs=1000, plot_metrics=True)

if __name__ == '__main__':
    train()

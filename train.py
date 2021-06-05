import pandas as pd

from models.baseline.encoder_decoder_trainer import EncoderDecoderTrainer as BahdanauAttention
from models.transformers.encoder_decoder_trainer import EncoderDecoderTrainer as Transformer


def train():
    # trainer = BahdanauAttention(batch_size=4)
    trainer = Transformer(batch_size=4)
    # trainer.train(
    #     data_csv_path='bms-molecular-translation/extended_data_1000.csv',
    #     num_epochs=1000, plot_metrics=True)

    # trainer.evaluate(
    #      data_csv_path='bms-molecular-translation/extended_data_4.csv',
    #      model_state_file='saved_models/transformer_model_state_0_levenshtein_1000_images_300_epochs.pth'
    # )

if __name__ == '__main__':
    train()

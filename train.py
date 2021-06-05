from models.baseline.encoder_decoder_operator import EncoderDecoderOperator as BahdanauAttention
from models.transformers.encoder_decoder_operator import EncoderDecoderOperator as Transformer


def train():
    trainer = BahdanauAttention(batch_size=4)
    # trainer = Transformer(batch_size=4)
    # trainer.train(
    #     data_csv_path='bms-molecular-translation/extended_data_30.csv',
    #     num_epochs=1000, plot_metrics=True)

    trainer.predict(
         data_csv_path='bms-molecular-translation/extended_data_30_test.csv',
         model_state_file='saved_models/seq2seq_rnn_model_state.pth'
    )

if __name__ == '__main__':
    train()

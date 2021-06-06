from argparse import ArgumentParser, Namespace

from models.baseline.encoder_decoder_operator import EncoderDecoderOperator as BahdanauAttention
from models.transformers.encoder_decoder_operator import EncoderDecoderOperator as Transformer
from models.visual_transformers.encoder_decoder_operator import EncoderDecoderOperator as VisualTransformer


# Old way of using
# TODO: Remove since it is depricated
def train():
    # trainer = Transformer(batch_size=4)
    trainer = BahdanauAttention(batch_size=4)

    # trainer.train(
    #     data_csv_path='bms-molecular-translation/extended_data_30.csv',
    #     num_epochs=1000, plot_metrics=True)

    trainer.predict(
         data_csv_path='bms-molecular-translation/extended_data_30_test.csv',
         model_state_file='saved_models/seq2seq_rnn_model_state.pth'
    )


def parse_command_args():
    parser = ArgumentParser()
    parser.add_argument("-df", "--data_file", dest="data_file", required=True,
                        help="specify the path to data stored in CSV format")
    parser.add_argument("-m", "--model", dest="model", required=True,
                        choices=["lstm", "transformer"],
                        help="specify the model")
    parser.add_argument("-a", "--action", dest="action", required=True,
                        choices=["train", "evaluate", "predict"],
                        help="specify an action")
    parser.add_argument("-bs", "--batch_size", dest="batch_size", required=False, default=4,
                        help="specify a batch size")
    parser.add_argument("-ne", "--num_epochs", dest="num_epochs", required=False, default=100,
                        help="specify the number of training epochs")
    parser.add_argument("-s", "--saved_model", dest="saved_model", required=False,
                        help="specify the path to a saved model")

    args = parser.parse_args()
    args.batch_size = int(args.batch_size)
    args.num_epochs = int(args.num_epochs)

    return args


def perform_action(args: Namespace):
    data_file_path = args.data_file
    model_name = args.model
    action = args.action
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model_state_file = args.saved_model

    if model_name == 'lstm':
        print('LSTM model chosen!')
        operator = BahdanauAttention(batch_size=batch_size)
        perform_model_action(operator, data_file_path, action, num_epochs, model_state_file)

    elif model_name == 'transformer':
        print('Transformer model chosen!')
        operator = Transformer(batch_size=batch_size)
        perform_model_action(operator, data_file_path, action, num_epochs, model_state_file)
    
    elif model_name == 'vis_transformer':
        print('Visual transformer model chosen!')
        operator = VisualTransformer(batch_size=batch_size)
        perform_model_action(operator, data_file_path, action, num_epochs, model_state_file)

    else:
        print("Invalid model specified")


def perform_model_action(operator, data_file_path: str, action: str, num_epochs=100, model_state_file=None):
    if action == 'train':
        print(f'Training for {num_epochs} epochs.')
        operator.train(
            data_csv_path=data_file_path,
            load_state_file=model_state_file,
            num_epochs=num_epochs, plot_metrics=True)
        return
    
    if model_state_file is None:
        print("Can not perform evaluation or prediction without a saved model specified!")
        return

    if action == 'evaluate':
        print(f'Evaluating model.')
        operator.evaluate(
            data_csv_path=data_file_path,
            model_state_file=model_state_file)

    elif action == 'predict':
        print(f'Predicting with model.')
        operator.predict(
            data_csv_path=data_file_path,
            model_state_file=model_state_file)


if __name__ == '__main__':
    args = parse_command_args()
    perform_action(args)

import pandas as pd

from models.baseline.encoder_decoder_trainer import EncoderDecoderTrainer
from models.EfficientNet.efficientEncoder_decoder_trainer import EfficientEncoderDecoderTrainer

EFFICIENT_MODELS = {
    'b0': { 'model_name': 'efficientnet-b0', 'encoder_dim': 1280 },
    'b1': { 'model_name': 'efficientnet-b1', 'encoder_dim':1280 },
    'b2': { 'model_name': 'efficientnet-b2', 'encoder_dim':1408 },
    'b3': { 'model_name': 'efficientnet-b3', 'encoder_dim':1536 }
}

def train():
    trainer = EncoderDecoderTrainer()
    dataframe = pd.read_csv("..\\bms-molecular-translation/extended_data_200.csv")
    trainer.train(dataframe, 1000, plot_metrics=True)

def trainEfficientNet():
    trainer = EfficientEncoderDecoderTrainer(model_name=EFFICIENT_MODELS['b0']['model_name'],
                                            encoder_dim=EFFICIENT_MODELS['b0']['encoder_dim'])
    dataframe = pd.read_csv("..\\bms-molecular-translation/extended_data_200.csv")
    trainer.train(dataframe, 1000, plot_metrics=True)

if __name__ == '__main__':
    train()

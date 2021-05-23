import torch.nn as nn
from efficientnet_pytorch import EfficientNet

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8'
)

class EfficientNetEncoder(nn.Module):
  def __init__(self, model_name):
    super(EfficientNetEncoder, self).__init__()

    if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    self.effNet = EfficientNet.from_name(model_name, in_channels=1, num_classes=2048, include_top=False)

  def forward(self, images):
    features = self.effNet(images)
    features = features.permute(0, 2, 3, 1)
    features = features.view(features.size(0), -1, features.size(-1))
    return features
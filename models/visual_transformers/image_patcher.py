import torch.nn as nn


class ImagePatcher(nn.Module):

    IMAGE_SIZE = 256
    IMAGE_FEATURE_SIZE = 2048
    PATCH_SIZE = 16

    def __init__(self):
        super(ImagePatcher, self).__init__()

    def forward(self, images):
        print(images.shape)
        features = None

        # shape = [batch_size, 256, 2048]
        return features

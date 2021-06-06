import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImagePatcher(nn.Module):

    NUM_CHANNELS = 1
    IMAGE_SIZE = 256
    IMAGE_FEATURE_SIZE = 256
    PATCH_SIZE = 16

    def __init__(self):
        super(ImagePatcher, self).__init__()

        if ImagePatcher.IMAGE_SIZE % ImagePatcher.PATCH_SIZE != 0:
            raise ValueError("IMAGE SIZE should be divisible by PATCH SIZE")
        
        self.unfold = nn.Unfold(
            kernel_size=(ImagePatcher.PATCH_SIZE, ImagePatcher.PATCH_SIZE),
            stride=ImagePatcher.PATCH_SIZE)

    def forward(self, images):
        # images shape [batch_size, num_channels, image_width, image_height]
        # features shape [batch_size, patch_size_flattened, num_patches(seq length)]
        features = self.unfold(images)
        
        # return shape [batch_size, sequence_length, feature_size]
        return features.transpose(1, 2)

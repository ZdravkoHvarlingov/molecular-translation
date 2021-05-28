import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for param in resnet.parameters():
            param.requires_grad_(True)
            
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))

        # shape = [batch_size, 64, 2048]



        return features

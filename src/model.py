import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LULCSegNet(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Encoder: remove fc e avgpool
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder simples
        self.decoder4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.encoder_conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        d4 = self.decoder4(x5)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        out = self.final_conv(d1)
        out = F.interpolate(
            out, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )
        return out

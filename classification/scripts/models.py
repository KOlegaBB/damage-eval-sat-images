import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------
# Utility Modules
# --------------------

class InitialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7,
                              stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MaskAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_feat, mask_feat):
        if mask_feat.shape[2:] != img_feat.shape[2:]:
            mask_feat = F.interpolate(mask_feat, size=img_feat.shape[2:],
                                      mode='bilinear', align_corners=False)
        attention = torch.sigmoid(mask_feat)
        return img_feat * attention


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# --------------------
# ResNet Siamese Network
# --------------------

def get_pretrained_resnet34(output_size, freeze=False):
    model_conv = models.resnet34(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, output_size)

    if not freeze:
        for name, child in model_conv.named_children():
            if name in ["layer3", "layer4"]:
                for param in child.parameters():
                    param.requires_grad = True

    return model_conv


class ResNetSiameseNetwork(nn.Module):
    def __init__(
            self,
            output_size=512,
            similarity_layers_sizes=[512, 512],
            dropout=0.5,
            output_type="classification",
            n_classes=4,
            freeze=False,
    ):
        """
        Construct the Siamese network with ResNet as feature extractor
        """
        super().__init__()
        self.left_network = get_pretrained_resnet34(output_size, freeze)
        self.right_network = get_pretrained_resnet34(output_size, freeze)

        similarity_layers = nn.Sequential(
            nn.Linear(output_size * 2, similarity_layers_sizes[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(similarity_layers_sizes[0]),
            nn.Dropout(dropout) if dropout else nn.Identity(),
        )

        prev_hidden_size = similarity_layers_sizes[0]
        for hidden in similarity_layers_sizes[1:]:
            similarity_layers.add_module("fc",
                                         nn.Linear(prev_hidden_size, hidden))
            similarity_layers.add_module("relu", nn.ReLU(inplace=True))
            similarity_layers.add_module("bn", nn.BatchNorm1d(hidden))
            similarity_layers.add_module("dropout", nn.Dropout(
                dropout) if dropout else nn.Identity())
            prev_hidden_size = hidden

        self.similarity = similarity_layers
        self.output = nn.Linear(hidden,
                                1) if output_type == "regression" else nn.Linear(
            hidden, n_classes)

    def forward(self, image_1, image_2):
        left_features = self.left_network(image_1)
        right_features = self.right_network(image_2)

        features = torch.cat([left_features, right_features], 1)
        sim_features = self.similarity(features)
        output = self.output(sim_features)
        return output


# --------------------
# DamageNet
# --------------------

class DamageNetStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = InitialConv(in_channels=3, out_channels=64)
        self.mask_conv = InitialConv(in_channels=1, out_channels=64)
        self.attention = MaskAttention()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, pre_img, post_img, mask):
        pre_feat = self.image_conv(pre_img)
        post_feat = self.image_conv(post_img)
        mask_feat = self.mask_conv(mask)

        pre_feat = self.attention(pre_feat, mask_feat)
        post_feat = self.attention(post_feat, mask_feat)

        pre_feat = self.pool(pre_feat)
        post_feat = self.pool(post_feat)

        return pre_feat, post_feat


class DamageNet(nn.Module):
    def __init__(self, mode='classification', n_classes=4):
        super().__init__()
        self.mode = mode
        self.n_classes = n_classes

        self.stem = DamageNetStem()
        self.res1 = ResBlock(64, 128)
        self.res2 = ResBlock(256, 128, stride=2)
        self.res3 = ResBlock(128, 256, stride=2)
        self.res4 = ResBlock(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1 if mode == 'regression' else n_classes)
        )

    def forward(self, pre_img, post_img, mask):
        pre_feat, post_feat = self.stem(pre_img, post_img, mask)
        pre_feat = self.res1(pre_feat)
        post_feat = self.res1(post_feat)
        x = torch.cat([pre_feat, post_feat], dim=1)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool(x)
        out = self.classifier(x)

        return out.squeeze(1) if self.mode == 'regression' else out

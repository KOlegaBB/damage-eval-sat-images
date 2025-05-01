import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------
# Utility Modules
# --------------------

class InitialConv(nn.Module):
    """
    Initial convolutional block with configurable kernel size, stride, and padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Default: 7.
        stride (int): Stride for the convolution. Default: 2.
        padding (int): Padding for the convolution. Default: 3.

    This block consists of:
    - A 2D convolution,
    - Batch normalization,
    - ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2,
                 padding=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (B, out_channels, H_out, W_out).
        """
        return self.relu(self.bn(self.conv(x)))


class Masking(nn.Module):
    """
    Applies a soft mask (via sigmoid) to image features.

    This module expects a mask feature map and applies it to the image feature map
    via element-wise multiplication after resizing and applying sigmoid.

    The purpose is to highlight important areas in the image features.

    Forward Inputs:
        img_feat (Tensor): Image features of shape (B, C, H, W).
        mask_feat (Tensor): Mask features of shape (B, 1, h, w).

    Returns:
        Tensor: Masked image features of shape (B, C, H, W).
    """

    def __init__(self):
        super().__init__()

    def forward(self, img_feat, mask_feat):
        if mask_feat.shape[2:] != img_feat.shape[2:]:
            mask_feat = F.interpolate(mask_feat, size=img_feat.shape[2:],
                                      mode='bilinear', align_corners=False)
        attention = torch.sigmoid(mask_feat)
        return img_feat * attention


class ResBlock(nn.Module):
    """
    A basic residual block with optional skip connection adjustment.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolution. Default is 1.

    The block structure:
        - Conv → BN → ReLU
        - Conv → BN
        - Residual connection (identity or projection)
        - Add & ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust the skip connection if shape changes (e.g., due to stride or channel mismatch)
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (B, out_channels, H_out, W_out).
        """
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# --------------------
# ResNet Siamese Network
# --------------------
# This implementation is adapted from the Siamese network architecture proposed in:
#
# Tinka Valentijn, Jacopo Margutti, Marc van den Homberg, and Jorma Laaksonen.
# “Multi-Hazard and Spatial Transferability of a CNN for Automated Building Damage Assessment.”
# *Remote Sensing*, 12(17), 2020.
# DOI: https://doi.org/10.3390/rs12172839
# Original codebase (Inception-based Siamese network) is available at:
# https://github.com/rodekruis/caladrius
#
# License: GNU General Public License v3.0 (GPLv3)
#
# This adaptation replaces the Inception-based backbone with ResNet-34
#
# The original license terms are preserved below:
#
# GNU GENERAL PUBLIC LICENSE
# Version 3, 29 June 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
# [License text truncated for brevity in this file. Full license at: https://www.gnu.org/licenses/gpl-3.0.html]


class ResNetSiameseNetwork(nn.Module):
    """
    Siamese neural network architecture using ResNet-34 as a shared or parallel feature extractor.
    Designed for tasks involving similarity learning, such as classification or regression on pairs of images.

    Attributes:
        left_network (nn.Module): ResNet-34 network to process the first image.
        right_network (nn.Module): ResNet-34 network to process the second image.
        similarity (nn.Sequential): Fully connected layers to learn similarity between feature vectors.
        output (nn.Linear): Final output layer, either for classification or regression.
    """

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
        Initializes the Siamese network.

        Args:
            output_size (int): Size of the output feature vector from each ResNet branch.
            similarity_layers_sizes (list): List of hidden layer sizes in the similarity head.
            dropout (float): Dropout probability in the similarity layers.
            output_type (str): Type of final output. Either "classification" or "regression".
            n_classes (int): Number of output classes (only used if output_type="classification").
            freeze (bool): If True, freeze all ResNet layers except layer3 and layer4.
        """
        super().__init__()
        self.left_network = self.get_pretrained_resnet34(output_size, freeze)
        self.right_network = self.get_pretrained_resnet34(output_size, freeze)

        # Build similarity head
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
        self.output = (
            nn.Linear(prev_hidden_size, 1)
            if output_type == "regression"
            else nn.Linear(prev_hidden_size, n_classes)
        )

    def forward(self, image_1, image_2):
        """
        Forward pass of the Siamese network.

        Args:
            image_1 (Tensor): First input image tensor of shape (B, C, H, W).
            image_2 (Tensor): Second input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, 1) for regression or (B, n_classes) for classification.
        """
        left_features = self.left_network(image_1)
        right_features = self.right_network(image_2)

        # Concatenate features and pass through similarity and output layers
        features = torch.cat([left_features, right_features], dim=1)
        sim_features = self.similarity(features)
        output = self.output(sim_features)
        return output

    @staticmethod
    def get_pretrained_resnet34(output_size, freeze=False):
        """
        Loads a pretrained ResNet-34 model and modifies the final fully connected layer.

        Args:
            output_size (int): Size of the output feature vector from ResNet.
            freeze (bool): If True, freeze all layers except layer3 and layer4.

        Returns:
            nn.Module: Modified ResNet-34 model.
        """
        model_conv = models.resnet34(pretrained=True)

        # Freeze all parameters by default
        for param in model_conv.parameters():
            param.requires_grad = False

        # Replace final fully connected layer
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, output_size)

        # Unfreeze selected layers if specified
        if not freeze:
            for name, child in model_conv.named_children():
                if name in ["layer3", "layer4"]:
                    for param in child.parameters():
                        param.requires_grad = True

        return model_conv


# --------------------
# DamageNet
# --------------------
# This implementation is based on the architecture described in:
# Isabelle Bouchard et al. “On Transfer Learning for Building Damage Assessment from
# Satellite Imagery in Emergency Contexts”. In: *Remote Sensing* 14.11 (2022).
# DOI: https://doi.org/10.3390/rs14112532

class DamageNetStem(nn.Module):
    """
    Stem block of the DamageNet model.
    Processes pre-disaster and post-disaster satellite images using shared convolutional layers,
    with an attention mechanism guided by a binary mask (e.g., building locations).

    Attributes:
        image_conv (nn.Module): Convolutional block for RGB images.
        mask_conv (nn.Module): Convolutional block for binary mask.
        attention (nn.Module): Attention mechanism that applies the mask to image features.
        pool (nn.Module): Max pooling to reduce spatial dimensions.
    """

    def __init__(self):
        super().__init__()
        self.image_conv = InitialConv(in_channels=3, out_channels=64)
        self.mask_conv = InitialConv(in_channels=1, out_channels=64)
        self.attention = Masking()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, pre_img, post_img, mask):
        """
        Forward pass through the stem.

        Args:
            pre_img (Tensor): Pre-disaster RGB image tensor of shape (B, 3, H, W).
            post_img (Tensor): Post-disaster RGB image tensor of shape (B, 3, H, W).
            mask (Tensor): Binary mask tensor of shape (B, 1, H, W).

        Returns:
            Tuple[Tensor, Tensor]: Feature maps for pre- and post-images after attention and pooling.
        """
        pre_feat = self.image_conv(pre_img)
        post_feat = self.image_conv(post_img)
        mask_feat = self.mask_conv(mask)

        pre_feat = self.attention(pre_feat, mask_feat)
        post_feat = self.attention(post_feat, mask_feat)

        pre_feat = self.pool(pre_feat)
        post_feat = self.pool(post_feat)

        return pre_feat, post_feat


class DamageNet(nn.Module):
    """
    DamageNet model for building damage assessment from satellite imagery.

    Architecture:
        - Dual stem for pre- and post-disaster image encoding
        - Residual blocks for hierarchical feature extraction
        - Feature fusion from pre/post branches
        - Final classification or regression head

    Args:
        mode (str): Task type, either 'classification' or 'regression'.
        n_classes (int): Number of output classes if classification.
    """

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
        """
        Forward pass of the DamageNet.

        Args:
            pre_img (Tensor): Pre-disaster RGB image tensor of shape (B, 3, H, W).
            post_img (Tensor): Post-disaster RGB image tensor of shape (B, 3, H, W).
            mask (Tensor): Binary mask tensor of shape (B, 1, H, W).

        Returns:
            Tensor: Output tensor of shape (B,) for regression or (B, n_classes) for classification.
        """
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

import torch
import torch.nn as nn

# --------------------
# Building Regularisation Network
# --------------------
# This implementation is adapted from the generator architecture used in:
#
# Zorzi, Stefano, Bittner, Ksenia, and Fraundorfer, Friedrich.
# "Machine-learned regularization and polygonization of building segmentation masks."
# *25th International Conference on Pattern Recognition (ICPR)*, 2021.
# DOI: https://doi.org/10.1109/ICPR48806.2021.9413197
#
# License: ICG Software - 2023, all rights reserved

class ResidualBlock(nn.Module):
    """
    Residual block consisting of two convolutional layers with InstanceNorm and ReLU.
    Used to maintain spatial resolution while learning transformation features.

    Args:
        in_features (int): Number of input and output channels.
    """

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1,
                      padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1,
                      padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Output feature map after residual addition.
        """
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """
    Generator network composed of residual blocks followed by upsampling layers.
    Converts a high-dimensional feature map into a 2-channel mask output.

    Args:
        num_residual_blocks (int): Number of residual blocks.
        in_features (int): Number of channels in the encoder output.
    """

    def __init__(self, num_residual_blocks=8, in_features=256):
        super(GeneratorResNet, self).__init__()

        out_features = in_features
        model = []

        # Residual blocks
        for _ in range(num_residual_blocks):
            model.append(ResidualBlock(out_features))

        # Upsampling blocks
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1,
                          padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, kernel_size=3, stride=1,
                          padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, kernel_size=3, stride=1,
                          padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer: 2-channel soft mask with sigmoid activation
        model += [
            nn.Conv2d(out_features, 2, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, feature_map):
        """
        Forward pass through the generator network.

        Args:
            feature_map (Tensor): Input tensor from encoder, shape (B, C, H, W).

        Returns:
            Tensor: 2-channel mask output of shape (B, 2, H', W').
        """
        return self.model(feature_map)


class Encoder(nn.Module):
    """
    Encoder network that downsamples and encodes the input images and masks.
    Suitable as a feature extractor for downstream generation tasks.

    Args:
        channels (int): Number of input channels. Default is 5 (e.g., 3 RGB + 2 masks).
    """

    def __init__(self, channels=5):
        super(Encoder, self).__init__()

        out_features = 64
        model = [
            nn.Conv2d(channels, out_features, kernel_size=7, stride=1,
                      padding=3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Two downsampling stages with double convolution and max pooling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1,
                          padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, kernel_size=3, stride=1,
                          padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_features = out_features

        self.model = nn.Sequential(*model)

    def forward(self, arguments):
        """
        Forward pass of the encoder.

        Args:
            arguments (List[Tensor]): List of tensors to be concatenated along the channel dimension.

        Returns:
            Tensor: Encoded feature map of shape (B, C, H/4, W/4).
        """
        x = torch.cat(arguments, dim=1)
        return self.model(x)


def regularize(images, outputs, encoder, generator, num_classes=2):
    """
    Applies regularization to the predictions using the encoder and generator.

    Args:
        images (torch.Tensor): The input batch of images.
        outputs (torch.Tensor): The output predictions from the model (logits).
        encoder (nn.Module): The encoder model used to generate latent features.
        generator (nn.Module): The generator model used to regularize the outputs.
        num_classes (int): Number of classes for the segmentation task.

    Returns:
        torch.Tensor: The regularized predictions after applying the generator.
    """

    def to_one_hot(labels, num_classes=2):
        """
        Convert class index labels to one-hot encoded format for semantic segmentation tasks.
        """
        batch_size, height, width = labels.shape
        one_hot = torch.zeros(batch_size, num_classes, height, width,
                              device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1),
                         255)  # Scatter the class indices into one-hot encoding
        return one_hot

    # Convert model outputs to one-hot encoding
    pred_masks_one_hot = to_one_hot(outputs.argmax(dim=1),
                                    num_classes=num_classes)

    # Pass through the encoder to get the latent features
    latent = encoder([images,
                      pred_masks_one_hot])  # Assuming encoder takes both images and one-hot encoded masks

    # Apply the generator to get the regularized outputs
    reg_output = generator(latent)

    return reg_output

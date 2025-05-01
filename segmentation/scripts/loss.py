import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class HybridLoss(nn.Module):
    """
    A custom loss function that combines Dice Loss and Cross-Entropy Loss
    for multiclass segmentation tasks.

    Args:
        None

    Attributes:
        dice_loss (smp.losses.DiceLoss): Dice loss implementation for multiclass segmentation.
        ce_loss (torch.nn.CrossEntropyLoss): Cross-entropy loss for multiclass classification.
    """
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        targets = torch.argmax(targets, dim=1)  # Convert one-hot encoded target to class indices
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets.long())
        return dice_loss + ce_loss

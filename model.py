import torchvision
import torch.nn as nn
import torch
from torchvision import models

def create_classifier(pretrained=True):
    """
    Creates a custom classifier based on the ResNet-18 architecture.

    Args:
        pretrained (bool, optional): If True, loads the pretrained ResNet-18 model weights.

    Returns:
        torch.nn.Module: Custom ResNet-18 model with modified fully connected layers.
    """
    # Load the ResNet-18 model
    model = models.resnet18(pretrained=pretrained)

    # Freeze the feature extraction layers if required
    # Uncomment the following lines to freeze the layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the architecture for classification
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Replace global average pooling with adaptive one
    model.fc = nn.Linear(512, 2)  # Replace the final fully connected layer

    # show the archietcture to the terminal
    print(model)
    
    return model
import torchvision.models as models
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

def vgg19():
    # Load VGG-19 pretrained model
    vgg19 = models.vgg19(pretrained=True)

    # Modify the first convolutional layer to accept 1 input channel
    vgg19.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Freeze all layers except the last three
    for param in list(vgg19.parameters())[:-3]:
        param.requires_grad = False

    # Customizing the last layer to match binary classification
    num_classes = 2  # number of classes for binary classification
    vgg19.classifier[-1] = nn.Linear(vgg19.classifier[-1].in_features, num_classes)

    # Move the model to the appropriate device
    vgg19 = vgg19.to(device)
    return vgg19
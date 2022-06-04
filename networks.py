from matplotlib.cbook import flatten
import torch
import torchvision
import torch.nn as nn


def weight_init(m):
  """
    Used as argument in model.apply to initialize Linear and Conv2d layers weights.
  """
  # if isinstance(m, nn.Linear):
  #   nn.init.xavier_uniform_(m.weight)
  #   nn.init.constant_(m.bias, 0.01)

  # elif isinstance(m, nn.Conv2d):
  #   nn.init.xavier_uniform_(m.weight)
  #   nn.init.zeros_(m.bias)

  classname = m.__class__.__name__
  if classname.find('Conv2d') != -1:
      torch.nn.init.xavier_uniform_(m.weight)
      torch.nn.init.zeros_(m.bias)
  elif classname.find('BatchNorm') != -1:
      m.weight.data.normal_(1.0, 0.01)
      m.bias.data.fill_(0)
  elif classname.find('Linear') != -1:
      m.weight.data.normal_(0.0, 0.01)
      m.bias.data.normal_(0.0, 0.01)


class FeatureExtractor(nn.Module):
  """
    CNN used to extract features
    - input: image samples;     shape=(batch, 3, 224, 224)
    - output: features;         shape=(batch, 512, 7, 7)
  """
  def __init__(self):
    super(FeatureExtractor, self).__init__()
    resnet = torchvision.models.resnet18(pretrained=True)
    # remove last 2 layers
    self.resnet_crop = nn.Sequential( *list(resnet.children())[:-2] )

    # for i, lay in enumerate(self.resnet.children()):
    #   print(i, ':', lay)
    #   print()

  def forward(self, x):
    return self.resnet_crop(x)


class RecognitionClassifier(nn.Module):
  """
    Classifier used to perform the recognition task.
    - input: output from FeatureExtractor;  shape=(batch, in_channels, 7, 7)
    - output: classes (0->out_classes);     shape=(batch)
  """
  def __init__(self, in_channels, out_classes):
    super(RecognitionClassifier, self).__init__()

    self.gap = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten()
    )

    self.fc1 = nn.Sequential(
      nn.Linear(in_channels, 1000),
      nn.BatchNorm1d(1000),
      nn.ReLU(),
      nn.Dropout(p=0.5)
    )
    
    self.fc2 = nn.Linear(in_features=1000, out_features=out_classes)


  def forward(self, x):
    x = torch.squeeze(self.gap(x)) # Nx512

    x1 = self.fc1(x)
    logits = self.fc2(x1)
    
    return logits



class RotationClassifier(nn.Module):
  """
    Classifier used to perform the relative rotation task.
    - input: output from FeatureExtractor; shape=(batch, in_channels, 7, 7)
    - output: classes for relative rotation (0->3); shape=(batch)
  """
  def __init__(self, in_channels, out_classes=4, hidden_size=100):
    super(RotationClassifier, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, hidden_size, (1, 1)),
      nn.BatchNorm2d(hidden_size),
      nn.ReLU()
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), stride=(2, 2)),
      nn.BatchNorm2d(hidden_size),
      nn.ReLU()
    )

    self.fc1 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(3*3*hidden_size, hidden_size),
      nn.BatchNorm1d(hidden_size, affine=True),
      nn.ReLU(),
      nn.Dropout(p=0.5)
    )

    self.fc2 = nn.Linear(hidden_size, out_classes)


  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.fc1(x)
    return self.fc2(x)

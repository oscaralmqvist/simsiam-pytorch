import torch
from torch import nn
import torchvision.models as models

class SimSiam(nn.Module):
  def __init__(self, d=1024):
    super().__init__()
    self.encoder = models.resnet18() # TODO Add option for ResNet-50
    enc_size = self.encoder.fc.out_features

    self.projection = nn.Sequential(
                          self.encoder,
                          nn.Linear(enc_size, 2048),
                          nn.BatchNorm1d(2048),
                          nn.ReLU(),
                          nn.Linear(2048, 2048),
                          nn.BatchNorm1d(2048),
                          nn.ReLU(),
                          nn.Linear(2048, d),
                          nn.BatchNorm1d(d)
                          )

    self.prediction = nn.Sequential(
                          nn.Linear(d, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(),
                          nn.Linear(512, d)
                          )

  def forward(self, x1, x2):
    z1 = self.projection(x1)
    z2 = self.projection(x2)

    p1 = self.prediction(z1)
    p2 = self.prediction(z2)

    return p1, p2, z1, z2

  def D(self, p, z):
    z = z.detach()

    p = torch.norm(p, dim=1)
    z = torch.norm(z, dim=1)

    return -(p*z).sum().mean()

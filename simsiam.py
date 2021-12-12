from torch import nn
import torchvision.models as models

import torch.nn.functional as F

class SimSiam(nn.Module):
  def __init__(self, d=2048, encoder=models.resnet50):
    super().__init__()
    self.encoder = encoder(pretrained=False)
    enc_size = 512#self.encoder.fc.in_features

    enc_modules = list(self.encoder.children())[:-1]
    self.embeddings = nn.Sequential(*enc_modules)

    self.projection = nn.Sequential(
                          #self.embeddings,
                          #nn.Squeeze(),
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

  def get_embedding(self, x):
    #return self.encoder(x).squeeze()
    return self.embeddings(x).squeeze()

  def forward(self, x1, x2):
    x1 = self.embeddings(x1).squeeze()
    x2 = self.embeddings(x2).squeeze()

    z1 = self.projection(x1)
    z2 = self.projection(x2)

    p1 = self.prediction(z1)
    p2 = self.prediction(z2)

    return p1, p2, z1, z2

  def D(self, p, z):
    z = z.detach()

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    return -(p*z).sum(dim=1).mean()

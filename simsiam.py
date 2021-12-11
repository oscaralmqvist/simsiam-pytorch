from torch import nn
import torchvision.models as models

import torch.nn.functional as F

class SimSiam(nn.Module):
  def __init__(self, d=2048, encoder=models.resnet50):
    super().__init__()
    self.encoder = encoder(pretrained=False, num_classes=d)
    enc_size = 512#self.encoder.fc.out_features

    enc_modules = list(self.encoder.children())[:-1]
    self.embeddings = nn.Sequential(*enc_modules)

    """
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
    """
    self.projection = self.encoder
    prev_dim = enc_size
    self.projection.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.projection.fc,
                                        nn.BatchNorm1d(d, affine=False)) # output layer
    self.projection.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

    """
    self.projection = nn.Sequential(
                          self.embeddings,
                          nn.Linear(enc_size, 2048),
                          nn.BatchNorm1d(2048),
                          nn.ReLU(),
                          nn.Linear(2048, 2048),
                          nn.BatchNorm1d(2048),
                          nn.ReLU(),
                          nn.Linear(2048, d),
                          nn.BatchNorm1d(d)
                          )
    """

    self.prediction = nn.Sequential(
                          nn.Linear(d, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(),
                          nn.Linear(512, d)
                          )

  def get_embedding(self, x):
    return self.embeddings(x).squeeze()

  def forward(self, x1, x2):
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

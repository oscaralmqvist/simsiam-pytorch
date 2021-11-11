import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn.functional as F

from simsiam import SimSiam

def get_augmentations(imgsize=64, crop=True, flip=True, jitter=True, grayscale=True, blur=True):
  augs = []

  if crop:
    augs.append(transforms.RandomResizedCrop(imgsize, scale=(0.2, 1.0)))
  if flip:
    # NOTE The original paper did not specify probability of flip, so we use the default 0.5
    augs.append(transforms.RandomHorizontalFlip())
  if jitter:
    augs.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8))
  if grayscale:
    augs.append(transforms.RandomGrayscale(p=0.2))
  if blur:
    # NOTE The original paper may not have used PyTorch for the gaussian blur
    # NOTE The kernel size was not specified in the paper. Thus we use 3
    augs.append(transforms.GaussianBlur(3, sigma=(0.1, 2.0)))

  return transforms.Compose(augs)

def validate(model, trainloader, validationloader):
  print('KNN')
  model.eval()
  xtrain = []
  ytrain = []
  with torch.no_grad():
    for i, batch in enumerate(trainloader):
      x, y = batch
      xtrain.append(F.normalize(model(x, x)[1], dim=1))
      ytrain.append(y)
      if i % 200 == 0:
        print(i)

    xtrain = torch.cat(xtrain, dim=0)
    ytrain = torch.cat(ytrain, dim=0)

    correct = 0
    total = 0

    for i, batch in enumerate(validationloader):
      x, y = batch
      z = F.normalize(model(x, x)[1], dim=1)

      if i % 50 == 0:
        print(i)

      dist = torch.cdist(xtrain, z, p=2)
      knn_pred = ytrain[dist.topk(1, largest=False, dim=0).indices].T[0]
      correct += torch.sum(knn_pred == y).item()
      total += x.shape[0]

  print('1NN accuracy: {}'.format(correct/total))

  model.train()

def main(datasetname):
  augmentations = get_augmentations()
  transform = transforms.ToTensor()
  if datasetname == 'CIFAR10':
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(
                                        root='./data/CIFAR10/',
                                        train=True,
                                        download=True,
                                        transform=transform)
    validationset = torchvision.datasets.CIFAR10(
                                        root='./data/CIFAR10/',
                                        train=False,
                                        download=True,
                                        transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=4)

  model = SimSiam()
  # TODO Add LR scheduler
  optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)

  smooth_loss = 0.0
  n_iter = 0
  for epoch in range(100):
    for i, batch in enumerate(trainloader, 0):
      x, y = batch
      n_iter += batch_size

      x1 = augmentations(x)
      x2 = augmentations(x)

      p1, p2, z1, z2 = model.forward(x1, x2)

      loss = 0.5*model.D(p1, z2) + 0.5*model.D(p2, z1)
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      lossval = loss.item()
      smooth_loss = smooth_loss*0.99 + lossval*0.01

      if i % 100 == 0:
        print('ITER: {}, loss: {}, smooth_loss: {}'.format(n_iter, lossval, smooth_loss))

    validate(model, trainloader, validationloader)

if __name__ == '__main__':
  main('CIFAR10')

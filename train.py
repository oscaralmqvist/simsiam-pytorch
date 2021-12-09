import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.optim as optim

import torch.nn.functional as F

from simsiam import SimSiam

import wandb

import argparse
import datetime

def get_augmentations(imgsize=64, crop=True, flip=True, jitter=True, grayscale=True, blur=True):
  augs = []
  augs_text = ''
  if crop:
    augs.append(transforms.RandomResizedCrop(imgsize, scale=(0.2, 1.0)))
    augs_text += 'crop,'
  if flip:
    # NOTE The original paper did not specify probability of flip, so we use the default 0.5
    augs.append(transforms.RandomHorizontalFlip())
    augs_text += 'flip,'
  if jitter:
    augs.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8))
    augs_text += 'jitter,'
  if grayscale:
    augs.append(transforms.RandomGrayscale(p=0.2))
    augs_text += 'grayscale,'
  if blur:
    # NOTE The original paper may not have used PyTorch for the gaussian blur
    # NOTE The kernel size was not specified in the paper. Thus we use 3
    augs.append(transforms.GaussianBlur(3, sigma=(0.1, 2.0)))
    augs_text += 'blur,'

  return transforms.Compose(augs), augs_text

def knn_validate(model, trainloader, validationloader, k=200):
  print('KNN')
  model.eval()
  xtrain = []
  ytrain = []
  with torch.no_grad():
    for i, batch in enumerate(trainloader):
      x, y = batch

      if torch.cuda.is_available():
        x, y = x.to('cuda'), y.to('cuda')

      xtrain.append(F.normalize(model.get_embedding(x), dim=1))
      ytrain.append(y)

    xtrain = torch.cat(xtrain, dim=0)
    ytrain = torch.cat(ytrain, dim=0)

    correct = 0
    total = 0

    for i, batch in enumerate(validationloader):
      x, y = batch

      if torch.cuda.is_available():
        x, y = x.to('cuda'), y.to('cuda')

      z = F.normalize(model.get_embedding(x), dim=1)
      dist = torch.cdist(xtrain, z, p=2)
      knn_pred, _ = torch.mode(ytrain[dist.topk(k, largest=False, dim=0).indices], dim=0)
      correct += torch.sum(knn_pred == y).item()
      total += x.shape[0]

  print('1NN accuracy: {}'.format(correct/total))

  model.train()

  return correct/total

def get_lr(optimizer):
  for p in optimizer.param_groups:
    return p['lr']

def get_momentum(optimizer):
  for p in optimizer.param_groups:
    return p['momentum']

def get_weight_decay(optimizer):
  for p in optimizer.param_groups:
    return p['weight_decay']


def main(datasetname, runname):
  print("using cuda?",torch.cuda.is_available())
  n_epochs = 100

  augs_text = ''
  batch_size = 512
  if datasetname == 'CIFAR10':
    n_epochs = 800
    model = SimSiam(encoder=models.resnet18)

    # As described in the paper, blur wasn't used for CIFAR10 experiments
    augmentations, augs_text = get_augmentations(imgsize=32, blur=False)
    transform = transforms.ToTensor() 

    dataset_config = {
      'root': './data/CIFAR10/',
      'download': True,
      'transform': transform
    }

    dataloader_config = {
      'batch_size': batch_size,
      'shuffle': True,
      'num_workers': 4,
      'pin_memory': torch.cuda.is_available()
    }

    trainset = torchvision.datasets.CIFAR10(train=True, **dataset_config)
    validationset = torchvision.datasets.CIFAR10(train=False, **dataset_config)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_config)
    validationloader = torch.utils.data.DataLoader(validationset, **dataloader_config)

    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

  if torch.cuda.is_available():
    model = model.to('cuda')

  config = {
    'dataset': datasetname,
    'n_epochs': n_epochs,
    'optimizer': type(optimizer),
    'lr': get_lr(optimizer),
    'momentum': get_momentum(optimizer),
    'weight_decay': get_weight_decay(optimizer),
    'batch_size': dataloader_config['batch_size'],
    'augmentations': augs_text
  }
  wandb.init(project='simsiam', entity='simsiambois', save_code=True, group='simsiambois', tags=datasetname, name=runname, config=config)

  wandb.watch(model)

  smooth_loss = 0.0
  n_iter = 0
  for epoch in range(n_epochs):
    print(f"epoch = {epoch}, smooth loss = {smooth_loss}")
    for i, batch in enumerate(trainloader, 0):
      x, _ = batch
      
      if torch.cuda.is_available():
        x = x.to('cuda')

      n_iter += dataloader_config['batch_size']

      x1 = augmentations(x)
      x2 = augmentations(x)

      p1, p2, z1, z2 = model.forward(x1, x2)

      loss = 0.5*model.D(p1, z2) + 0.5*model.D(p2, z1)
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      lossval = loss.item()
      smooth_loss = smooth_loss*0.99 + lossval*0.01

      wandb.log({'n_iter': n_iter, 'epoch': epoch, 'loss': lossval})

      if i % 100 == 0 and i != 0:
        print('ITER: {}, loss: {}, smooth_loss: {}'.format(n_iter, lossval, smooth_loss))

    if datasetname == 'CIFAR10':
      knn_acc = knn_validate(model, trainloader, validationloader)
      wandb.log({'n_iter': n_iter, 'epoch': epoch, '1nn_accuracy': knn_acc})

if __name__ == '__main__':
  # TODO Maybe add argparse stuff here
  main('CIFAR10', 'CIFAR10-testrun')

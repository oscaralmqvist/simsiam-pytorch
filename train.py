import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.optim as optim

import torch.nn.functional as F
import torch.nn as nn

from simsiam import SimSiam

import wandb

import argparse
import datetime

import math

def get_augmentations(imgsize=64, crop=True, flip=True, jitter=True, grayscale=True, blur=True):
  augs = []
  augs_text = ''
  if crop:
    augs.append(transforms.RandomResizedCrop(imgsize, scale=(0.2, 1.0)))
    augs_text += 'crop,'
  if flip:
    # NOTE The original paper did not specify probability of flip, so we use the default 0.5
    augs.append(transforms.RandomHorizontalFlip(p=0.5))
    augs_text += 'flip,'
  if jitter:
    augs.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8))
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

def linear_evaluate(model, trainloader, validationloader):
  linear = nn.Linear(2048, 10)
  linear.to('cuda')
  eval_loss = nn.CrossEntropyLoss()
  opt = optim.SGD(linear.parameters(), lr=30.0, momentum=0.9, weight_decay=0.0005)
  for ep in range(90):
    for i, batch in enumerate(trainloader, 0):
        x, y = batch
        if torch.cuda.is_available():
          x, y = x.to('cuda'), y.to('cuda')
        with torch.no_grad():
          z = model.get_embedding(x)
        yhat = linear(z)
        loss = eval_loss(yhat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

  correct = 0
  total = 0
  with torch.no_grad():
    for i, batch in enumerate(trainloader, 0):
        x, y = batch
        if torch.cuda.is_available():
          x, y = x.to('cuda'), y.to('cuda')

        z = model.get_embedding(x)
        yhat = torch.argmax(linear(z), dim=1)
        correct += torch.sum(yhat == y).item()
        total += x.shape[0]

  print('lin eval done: acc: {}'.format(correct/total))
  return correct/total

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

      votes = torch.zeros(x.shape[0], 10) # TODO Fix to be th number of classes instead of 10
      if torch.cuda.is_available():
        x, y = x.to('cuda'), y.to('cuda')
        votes = votes.to('cuda')

      z = F.normalize(model.get_embedding(x), dim=1)
      knn_pred = knn_predict(z, xtrain.T, ytrain, 10, k, 0.2)
      correct += torch.sum(knn_pred == y).item()
      total += x.shape[0]

  print('1NN accuracy: {}'.format(correct/total))

  model.train()

  return correct/total

# Code taken from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    #pred_labels = pred_scores.argsort(dim=-1, descending=True)
    pred_labels = pred_scores.argmax(dim=-1)

    return pred_labels

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
    augmentations, augs_text = get_augmentations(blur=False, imgsize=32)
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
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_config)

    validationset = torchvision.datasets.CIFAR10(train=False, **dataset_config)
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
    #knn_acc = knn_validate(model, trainloader, validationloader)
    #linear_evaluate(model, trainloader, validationloader)
    adjust_learning_rate(optimizer, 0.03, epoch, n_epochs)
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
      if epoch % 100 == 0:
        res = linear_evaluate(model, trainloader, validationloader)
        wandb.log({'n_iter': n_iter, 'linear_eval_acc': res})

def adjust_learning_rate(optimizer, init_lr, epoch, n_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / n_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr



if __name__ == '__main__':
  # TODO Maybe add argparse stuff here
  main('CIFAR10', 'CIFAR10-testrun')

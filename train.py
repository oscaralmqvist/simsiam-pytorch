import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

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

      if torch.cuda.is_available():
        x, y = x.to('cuda'), y.to('cuda')

      xtrain.append(F.normalize(model(x, x)[2], dim=1))
      ytrain.append(y)

    xtrain = torch.cat(xtrain, dim=0)
    ytrain = torch.cat(ytrain, dim=0)

    correct = 0
    total = 0

    for i, batch in enumerate(validationloader):
      x, y = batch

      if torch.cuda.is_available():
        x, y = x.to('cuda'), y.to('cuda')

      z = F.normalize(model(x, x)[2], dim=1)

      dist = torch.cdist(xtrain, z, p=2)
      knn_pred = ytrain[dist.topk(1, largest=False, dim=0).indices][0]
      correct += torch.sum(knn_pred == y).item()
      total += x.shape[0]

  print('1NN accuracy: {}'.format(correct/total))

  model.train()

def main(datasetname):
  print("using cuda?",torch.cuda.is_available())

  if datasetname == 'CIFAR10':
    n_epochs = 800
    model = SimSiam(encoder=models.resnet18)

    # As described in the paper, blur wasn't used for CIFAR10 experiments
    augmentations = get_augmentations(blur=False) 
    transform = transforms.ToTensor() 

    dataset_config = {
      'root': './data/CIFAR10/',
      'download': True,
      'transform': transform
    }

    dataloader_config = {
      'batch_size': 512,
      'shuffle': True,
      'num_workers': 4,
      'pin_memory': torch.cuda.is_available()
    }

    trainset = torchvision.datasets.CIFAR10(train=True, **dataset_config)
    validationset = torchvision.datasets.CIFAR10(train=False, **dataset_config)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_config)
    validationloader = torch.utils.data.DataLoader(validationset, **dataloader_config)

    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs) 

  if torch.cuda.is_available():
    model = model.to('cuda')

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

      if i % 100 == 0 and i != 0:
        print('ITER: {}, loss: {}, smooth_loss: {}'.format(n_iter, lossval, smooth_loss))

    scheduler.step()
    validate(model, trainloader, validationloader)

if __name__ == '__main__':
  main('CIFAR10')

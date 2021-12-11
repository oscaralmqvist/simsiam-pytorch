# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import os
import json
from PIL import Image
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        dataset_type = 'train' if train else 'val'

        self.transform = transform
        self.annotation_dir = os.path.join(root, 'annotations')
        self.img_dir = os.path.join(root, f'{dataset_type}2017')

        with open(os.path.join(self.annotation_dir, f'instances_{dataset_type}2017.json')) as f:
            self.coco_annotations = json.load(f)

    def __len__(self):
        return len(self.coco_annotations['images'])
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.coco_annotations['images'][idx]['file_name']) 

        sample = Image.open(img_path).convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        return sample

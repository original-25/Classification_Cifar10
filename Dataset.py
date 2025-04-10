import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, RandomAffine, ColorJitter
class MyCifarDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.transform = transform
        self.all_images=[]
        self.all_labels=[]

        folder_paths=[]
        if train:
            folder_paths = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            folder_paths = [os.path.join(root, "test_batch")]

        for path in folder_paths:
            with open(path, "rb") as fo:
                res = pickle.load(fo, encoding="latin1")
                self.all_images.extend(res["data"])
                self.all_labels.extend(res["labels"])


    def __len__(self):
        return len(self.all_labels)
    def __getitem__(self, index):
        image=self.all_images[index]
        label=self.all_labels[index]
        image = image.astype(np.float32)
        image = image.reshape(3, 32, 32)/255
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image, label
if __name__=="__main__":
    categories=["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    dataset = MyCifarDataset(root='../cifar/cifar-10-batches-py', train=True)
    count = {}
    for key in range(10):
        count[key]=0
    for i in range(dataset.__len__()):
        image, label = dataset.__getitem__(i)
        count[label]+=1

    print(count)
# getdata.py
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Default input image size

IMAGE_H = 100
IMAGE_W = 100

# Define a conversion relationship to convert image data into PyTorch's Tensor form
data_transform = transforms.Compose([
    transforms.ToTensor()   # Convert to Tensor form
])


class DogsVSCatsDataset(data.Dataset):      # Create a new data set class and inherit the data.Dataset parent class in PyTorch
    def __init__(self, mode, dir):
        self.mode = mode
        self.list_img = []                  # Create a new image list to store the image path
        self.list_label = []                # Create a label list to store the labels corresponding to the pictures
        self.data_size = 0                  # Record data set size
        self.transform = data_transform

        if self.mode == 'train':            # In the training set mode, the path and label of the image need to be extracted
            dir = dir + '/train1/'           # training path in"dir"/train/
            for file in os.listdir(dir):    # read dir
                self.list_img.append(dir + file)        # Add the image path and file name to the image list
                self.data_size += 1                     # Data set increased by 1
                name = file.split(sep='.')              # Split file name

                if name[0] == 'elevator':
                    self.list_label.append(0)         # If the picture is an elevator, the label is 0
                else:
                    self.list_label.append(1)         # If the picture is a wall , the label is 1
        elif self.mode == 'test':           # Extract image path in test set mode
            dir = dir + '/test/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)    # Add image path to image list
                self.data_size += 1
                self.list_label.append(2)
        else:
            return print('Undefined Dataset!')

    def __getitem__(self, item):            # Overload the data.Dataset parent class method to get the data content in the dataset
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))                        # Resize the image to a uniform size
            img = np.array(img)[:, :, :3]                               # Convert data into numpy array form
            label = self.list_label[item]                               # Get the label corresponding to the image
            return self.transform(img), torch.LongTensor([label])       # Convert image and label into PyTorch form and return
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))
            img = np.array(img)[:, :, :3]
            return self.transform(img)
            print('None')

    def __len__(self):
        return self.data_size


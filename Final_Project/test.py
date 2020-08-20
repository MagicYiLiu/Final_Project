
from Final_Project.getdata import DogsVSCatsDataset as DVCD
from Final_Project.network import Net
import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


dataset_dir = './data/'                 # dataset path
model_file = './model/model.pth'        # Model saved path

def test():

    model = Net()
    model.cuda()                                        # Computing with GPU
    model.load_state_dict(torch.load(model_file))       # Load the trained model parameters
    model.eval()                                         # Set to evaluation mode, that is, do not dropout during calculation

    datafile = DVCD('test', dataset_dir)                # Instantiate a data set
    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    index = np.random.randint(0, datafile.data_size, 1)[0]      # Get a random number, randomly get a test picture from the data set
    img = datafile.__getitem__(index)                           # get a image
    img = img.unsqueeze(0)                                      # Add a dimension to the acquired image data
    img = Variable(img).cuda()                                  # Place the data in the Variable node of PyTorch and send it to the GPU
                                                                # as the starting point for network calculation
    out = model(img)
    print(out)
    if out[0, 0] > out[0, 1]:
        print('the image is a cat')
    else:
        print('the image is a dog')

    img = Image.open(datafile.list_img[index])      # open test picture
    plt.figure('image')                             # display image
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test()


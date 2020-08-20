from Final_Project.getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from Final_Project.network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = './data/'             # data path

model_cp = './model/'               # Network parameter save location
workers = 10                        # Number of threads to read data
batch_size = 16                     # batch_size
lr = 0.0001                         # learning rate
nepoch = 1


def train():
    datafile = DVCD('train', dataset_dir)
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    # Use PyTorch's DataLoader class to encapsulate, to achieve data set sequence disorder, multi-threaded reading,
    # and multiple data at a time

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = Net()
    # model = model.cuda()                # use GPU
    model = nn.DataParallel(model)
    model.train()                       # The network is set to training mode

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Instantiate an optimizer, adjust the network
    # parameters, the optimization method is adam method

    criterion = torch.nn.CrossEntropyLoss()               # Define the loss calculation method

    cnt = 0             # Number of training images
    for epoch in range(nepoch):

        # Read the data in the dataset for training,
        for img, label in dataloader:
           # img, label = Variable(img).cuda(), Variable(label).cuda()           # Place the data in the Variable node of PyTorch and send it to the GPU as the starting point for network calculation
            out = model(img)                                                    # Calculate network output value
            loss = criterion(out, label.squeeze())      # Calculate the loss
            loss.backward()                             # Error back propagation
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))          # Print a batch size training result

    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))            # After training all the data, save the network parameters


if __name__ == '__main__':
    train()











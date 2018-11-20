'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset  # For custom datasets
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import argparse

from models import *
#from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Check if there is an operation
        some_operation = self.operation_arr[index]
        # If there is an operation
        if some_operation:
            # Do some operation on image
            # ...
            # ...
            pass
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


#Load training / testing data / validate
custom_dset_from_images =  \
    CustomDatasetFromImages('data/train.csv')

trainloader = torch.utils.data.DataLoader(dataset=custom_dset_from_images,
                                                batch_size=12,
                                                shuffle=False)
custom_dset_from_images =  \
    CustomDatasetFromImages('data/test.csv')

testloader  = torch.utils.data.DataLoader(dataset=custom_dset_from_images,
                                                batch_size=2,
                                                shuffle=False)

custom_dset_from_images =  \
    CustomDatasetFromImages('data/validate.csv')

validateloader  = torch.utils.data.DataLoader(dataset=custom_dset_from_images,
                                                batch_size=1,
                                                shuffle=False)



# Model
print('==> Building model..')
#
net = VGG('VGG19')
#net = DSK2(1)
#net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
#net = DenseNet121()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# dsk> Avoid CrossEntropy
criterion = nn.CrossEntropyLoss()
# dsk> Regression Loss
#criterion = nn.SmoothL1Loss()
#criterion = nn.KLDivLoss(size_average=False)
#criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print(len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        ##print(trainloader)
        outputs = net(inputs)
        ##print(inputs)
        ##print(outputs.type())
        ##print(outputs)
        if device == 'cuda':
            targets=targets.type(torch.cuda.FloatTensor) # CAST CUDA
        if device == 'cpu':
            targets=targets.type(torch.FloatTensor) #cast

        # Make sure tensors are same size for criterion loss calc
        dsk_targets=targets.view(outputs.size(0),1)
        loss = criterion(outputs, dsk_targets)
        loss.backward()
        optimizer.step()

        # Stop criterion
        if loss < 1e-3  :
            break
        print('Train Loss: {:.6f} after {} batches'.format(loss, batch_idx))


# Testting
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
	        # dsk> again
            if device == 'cuda':
                targets=targets.type(torch.cuda.FloatTensor) # CAST CUDA
            if device == 'cpu':
                targets=targets.type(torch.FloatTensor) #cast

            # Make sure tensors are same size for criterion loss calc
            dsk_targets=targets.view(outputs.size(0),1)
            loss = criterion(outputs, dsk_targets)
            print('Test Loss: {:.6f} after {} batches'.format(loss, batch_idx))
            # Stop criterion
            if loss < 1e-3:
                break

# save me
def save_model(epoch_toSave,diff):
    print('Saving..')
    diff_txt = '{:.2f}'.format(diff) 
    state = {
        'net': net.state_dict(),
 #       'acc': acc,
        'epoch': epoch_toSave,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+diff_txt+'_ckpt_'+str(epoch_toSave)+'.t7')
 #   best_acc = acc

def validate():
    global best_acc
    net.eval()
    validate_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validateloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
	    # dsk> again
            diff =  targets.tolist()[0] - outputs.tolist()[0][0]
            #diff = diff / targets.tolist()[0] * 100
              
    return diff


# Big Validation (105 32x32 per day 8000+ stocks)
"""
#checkpoint = torch.load('./checkpoint/0.19_ckpt_14.t7')
checkpoint = torch.load('./checkpoint/ckpt_11.t7')
net.load_state_dict(checkpoint['net'])

custom_dset_from_images =  \
    CustomDatasetFromImages('data/weighted_centroid_0.9_train.csv')
validateloader = torch.utils.data.DataLoader(dataset=custom_dset_from_images,
                                                batch_size=128,
                                                shuffle=False)
validate()

"""


# Quentin training style !
nb_run = 0
#make it simple
nep = 1
for nb_run in range(0,nep):
    print('Run '+str(nb_run))
    epoch = 0
    net = VGG('VGG19')
    net = net.to(device) 
    if device == 'cuda':
    	net = torch.nn.DataParallel(net)
    	cudnn.benchmark = True
    # change for crossentropy when using labels
    criterion = nn.KLDivLoss(size_average=False)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch, start_epoch+1+20):
        train(epoch)
        test(epoch)
        diff = validate()
        if (diff < 0):
            print('Exit: diff {:.1f}%'.format(diff))
            break
        if (diff < 2 ):
            save_model(epoch,diff)
            print('Save: diff {:.1f}%'.format(diff))
            
    #if (epoch % 11 == 0):
        #validate()
    #    save_model(epoch)


# BASURA

# plt.plot(x_train, y_correct, 'go', label = 'from data', alpha = .5)
# plt.plot(x_train, predicted, label = 'prediction', alpha = 0.5)
# plt.legend()
# plt.show()
# plt.savefig("output.png", bbox_inches='tight')

#checkpoint = torch.load('./checkpoint/ckpt.t7')
#net.load_state_dict(checkpoint['net'])

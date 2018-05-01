import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob
import os
import math
import random
import argparse
from bits import bit_flip

BATCH_SIZE = 64
variables = ['dens']
CONV_INPUT_SHAPE = (len(variables), 60, 60)


class FlashDataset(torch.utils.data.Dataset):
    def __init__(self, clean_data, error_data):
        self.data = []
        self.targets = []

        # Add error data
        self.data.append( error_data )
        self.targets.append(np.ones((error_data.shape[0], 1)))

        # Add clean data
        self.data.append( clean_data[0:error_data.shape[0]] )
        self.targets.append(np.zeros((error_data.shape[0], 1)))

        self.data = np.vstack(self.data)
        self.targets = np.vstack(self.targets)
        self.data = torch.from_numpy(self.data)
        self.targets = torch.from_numpy(self.targets)
        print "data size:", self.data.size(), ", targets size: ", self.targets.size()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.size(0)


class FlashNet(nn.Module):
    def __init__(self):
        super(FlashNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(len(variables), 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),

            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),

            nn.Conv2d(64, 96, 3, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1)
        )
        conv_output_size = self.get_conv_output_size()
        print "conv output size: ", conv_output_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=768, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(x.size(0), -1)   # flatten
        return self.fc.forward(x)

    # Helper function: find out the output size of conv layer
    # So we can pass it to the linear layer
    def get_conv_output_size(self):
        conv_input = Variable(torch.rand(BATCH_SIZE, *CONV_INPUT_SHAPE))
        conv_output = self.conv.forward(conv_input)
        output_size = conv_output.data.view(BATCH_SIZE, -1).size(1)
        return output_size

def training(model, train_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6, momentum=0.5)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    loss_func = nn.BCELoss()

    running_loss = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.data[0]
            if i % 50 == 0:
                print("epoch:%s i:%s loss:%s" %(epoch, i, running_loss/50))
                running_loss = 0

def evaluating(model, test_loader):
    num_correct = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)

        pred = (output.data >= 0.5).view(-1, 1)
        truth = (labels.data >= 0.5).view(-1, 1)
        num_correct += (pred == truth).sum()
        false_positive += ((pred^truth) & pred).sum()
        false_negative += ((pred^truth) & truth).sum()

        if i%50==0:
            print i, num_correct, false_positive, false_negative

    acc = num_correct / len(test_loader.dataset)
    fp = false_positive / len(test_loader.dataset)
    fn = false_negative / len(test_loader.dataset)
    print("acc: %s fp: %s fn: %s" %(acc, fp, fn))


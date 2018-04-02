import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob
import os
import math
import random
from bits import bit_flip

BATCH_SIZE = 64
CONV_INPUT_SHAPE = (3, 60, 60)
variables = ['dens', 'temp', 'pres']

def get_flip_error(val):
    while True:
        pos = random.randint(1, 20)
        error =  bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    error = min(10e+3, error)
    error = max(-10e+3, error)
    return error


class SedovDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, error_data_file=None):
        self.data = []
        self.targets = []

        # Read corrupted data
        #for filename in glob.iglob(data_dir+"error/data_11_bit0_20_iter20.npy"):
        for filename in glob.iglob(error_data_file):
            d = np.load(filename)
            print "read error data:", filename, d.shape
            '''
            for i in range(len(d)):
                x = random.randint(20, 40)
                y = random.randint(20, 40)
                d[i, 0, x, y] = get_flip_error(d[i, 0, x, y])
            '''
            self.data.append( d )
            self.targets.append(np.ones((d.shape[0], 1)))
        error_len = self.data[0].shape[0]

        # Read clean data
        for filename in glob.iglob(data_dir+"clean/data_11.npy"):
            d = np.load(filename)[0:error_len]
            #d = np.load(filename)
            print "read clean data:", filename, d.shape
            self.data.append( d )
            self.targets.append(np.zeros((d.shape[0], 1)))

        self.data = np.vstack(self.data)
        self.targets = np.vstack(self.targets)
        self.data = torch.from_numpy(self.data)
        self.targets = torch.from_numpy(self.targets)
        print "data size:", self.data.size(), ", targets size: ", self.targets.size()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.data.size(0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(6, 6), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
        )
        conv_output_size = self.get_conv_output_size()
        print "conv output size: ", conv_output_size
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=1, bias=True),
            #nn.ReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(in_features=4096, out_features=4096, bias=True),
            #nn.ReLU(),
            #nn.Linear(in_features=4096, out_features=1, bias=True),
            nn.Sigmoid(),
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

def training(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6, momentum=0.5)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    loss_func = nn.BCELoss()

    running_loss = 0
    for epoch in range(5):
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


if __name__ == "__main__":
    model_file = "./alex.model"
    model = None
    if os.path.isfile(model_file):
        print "load"
        model = torch.load(model_file)
    else:
        model = CNN().double()
        if torch.cuda.is_available():
            print "Have CUDA!!!"
            model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "More than one GPU card!!!"
            model = nn.DataParallel(model)

    print model

    #trainset = SedovDataset('/home/chenw/sources/train/')
    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    #testset = SedovDataset('/home/chenw/sources/train/')
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    #training(model, train_loader)
    #torch.save(model, model_file)
    #evaluating(model, train_loader)

    for error_data_file in glob.iglob("/home/chenw/sources/test/error3/data_11_0to20*.npy"):
        testset = SedovDataset('/home/chenw/sources/test/', error_data_file)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
        evaluating(model, test_loader)

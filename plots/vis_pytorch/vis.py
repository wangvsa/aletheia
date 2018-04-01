import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from alex import CNN

def save_weights_picture(weights, allkernels=False, nrow=8):
    n,c,w,h = weights.shape
    if allkernels: weights = weights.view(n*c,-1,w,h )
    print("weights:", weights.shape)
    utils.save_image(weights, "weights.png", nrow=nrow)
    return None

def vis_weights(weights, allkernels=False, nrow=8):

    n,c,w,h = weights.shape
    if allkernels: weights = weights.view(n*c,-1,w,h )
    elif c != 3: weights = weights[:,0,:,:].unsqueeze(dim=1)

    grid = utils.make_grid(weights, nrow=nrow, normalize=True, padding=2)
    #plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

    cmap = plt.cm.jet
    image = cmap(grid.cpu().numpy()[0])
    plt.imsave('weights.png', image)



if __name__ == "__main__":
    #alexnet = torchvision.models.alexnet(pretrained=True)
    #print(alexnet)
    model = torch.load("./alex.model")
    print model

    # weights of the first layer
    weights = model.module.conv[0].weight.data.clone()
    print(weights.shape)

    #save_weights_picture(weights, allkernels=True)
    vis_weights(weights, allkernels=False)

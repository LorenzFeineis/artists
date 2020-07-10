import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.datasets as datasets
import torch.utils.data as  torchdata
import torchvision.transforms as transforms

from NeuralNet import Net

test_image = Image.open("../Vincent_van_Gogh_171.jpg", "r")


#transform = transforms.Compose([transforms.FiveCrop(size = (256,256)),
#                                transforms.ToTensor() ])
x = transforms.Compose([transforms.CenterCrop((256,256)), transforms.ToTensor()])(test_image)
x = x[None,:,:,:]

test_image.close()

print(x.shape)
N, C, width, height = x.shape
y = torch.tensor([1])


net = Net(size= (width,height), num_classes = 2)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)

for epoch in range(30):
    optimizer.zero_grad()
    net.train()
    output = net(x)
    training_loss = loss(output, y)
    print(training_loss)
    training_loss.backward()
    optimizer.step()

print(net(x))

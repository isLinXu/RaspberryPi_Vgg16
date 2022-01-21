import mydatasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torch

train_dataset = mydatasets.CIFAR10_IMG('/media/hxzh/CC06287006285DA8/python/lib/Img_Aug-master/pytorch_learn/data', train=True, transform=transforms.ToTensor())
test_dataset = mydatasets.CIFAR10_IMG('/media/hxzh/CC06287006285DA8/python/lib/Img_Aug-master/pytorch_learn/data', train=False, transform=transforms.ToTensor())

# train_dataset = datasets.CIFAR10_IMG('./datasets',train=True)
# print(train_dataset.filenames)
# print(train_dataset.labels)
# img,label= train_dataset.__getitem__(1)
# img = np.transpose(img.numpy(),(1,2,0))
# plt.imshow(img)
# plt.show()
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


for step, (b_x, b_y) in enumerate(train_loader):
    if step < 1:
        imgs = torchvision.utils.make_grid(b_x)
        print(imgs.shape)
        imgs = np.transpose(imgs, (1, 2, 0))
        print(imgs.shape)
        plt.imshow(imgs)
        plt.show()
import torch
import torchvision
from torchvision import models
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from vocab import vocab
import matplotlib.pyplot as plt

def save_img(img_name, img):
	img_np = img.to('cpu').detach().numpy().copy()
	plt.imsave(img_name, img_np[0])

vocab.preprocess_caption()

img_size = (299,299)

model = models.resnet50(pretrained=True)
model.fc = nn.Identity()

trans = transforms.Compose([transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])

trainset = dset.ImageFolder(root='data/train', transform=trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 2048, shuffle = True, num_workers = 4)
trainsize = len(trainset)

valset = dset.ImageFolder(root='data/val', transform=trans)
valloader = torch.utils.data.DataLoader(valset, batch_size = 2048, shuffle = True, num_workers = 4)
valsize = len(valset)

class_names = trainset.classes
print(trainsize, valsize, class_names)

print(trainloader)
for data,label in trainloader:
	break
print(data)
print(label)

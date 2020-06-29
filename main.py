import torch
import torchvision
from torchvision import models
from torch import nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from vocab import vocab
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_img(img_name, img):
	img_np = img.to('cpu').detach().numpy().copy()
	plt.imsave(img_name, img_np[0])

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running in "+dev+".")
device = torch.device(dev)
BATCH_SIZE = 512

vocab.preprocess_caption(True)

img_size = (224,224)

model = models.resnet50(pretrained=True)
model.fc = nn.Identity()

trans = transforms.Compose([transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])

#trainset = dset.ImageFolder(root='data/train', transform=trans)
trainset = dset.CocoCaptions(root = './data/train/images',
                             annFile = './data/train/captions_train2014.json',
                             transform=trans)

target = np.array([])
label = np.array([])
with tqdm(range(100)) as pbar:
	pbar.set_description("[Data Converting]: ")
	for i in pbar:
		img, sent = trainset[i]
		target = np.append(target, model(torch.unsqueeze(img, 0)).to('cpu').detach().numpy().copy())
		label = np.append(label, vocab.tokenize_caption(sent))
target = torch.from_numpy(target).clone()
label = torch.from_numpy(label).clone()
# trainset = torch.utils.data.TensorDataset(label,target)
# #print(type(label), type(target))
# #trainloader = DataLoader([label, target], batch_size = 2048, shuffle = True) 
# trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
# trainsize = len(trainset)

# valset = dset.ImageFolder(root='data/val', transform=trans)
# valloader = torch.utils.data.DataLoader(valset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
# valsize = len(valset)

# print('Number of samples: ', len(trainset))
# img, target = trainset[3]
# vocab.tokenize_caption(target)
# #save_img("bbb.jpg", img)
# print('Image size: ', img.size())
# print(trainloader)

# for epoch in range(100):
# 	for label, inputs in trainloader:
# 		print("bbb")
# 		print(inputs)
# 		inputs = inputs.to(device)
# 		model(inputs)
# 		print("aaa")

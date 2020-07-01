import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from pycocotools.coco import COCO
import pickle
from PIL import Image
from nltk import tokenize
import re
import random

def COCO_loader(BATCH_SIZE,WORD_TO_ID,CAPTION_PATH,TRAIN_IMAGE_PATH):

    def tokenize_caption(caption):
        # Transform abbreviation
        caption = re.sub(r'\'d'," had", caption)
        caption = re.sub(r'\'m'," am", caption)
        caption = re.sub(r'\'s'," is", caption)
        caption = re.sub(r'[&]+'," and ", caption)
        # Remove special characters
        caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)
        tokens = tokenize.word_tokenize(caption.lower())

        sent = ['<start>'] + tokens + ['<end>']
        sent_id = [WORD_TO_ID[t] if (t in WORD_TO_ID) else WORD_TO_ID['<unk>'] for t in sent]
        return torch.Tensor(sent_id)

    def collate_fn(data):
        images, captions = zip(*data)
        # Choose captions randomly if there are several captions to one image, and tokenize
        captions = [tokenize_caption(c[random.randrange(len(c))]) for c in captions]
        data = zip(images, captions)
        # Sort data according to the length of captions
        data = sorted(data, key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Add 1D to make 4D tensor
        images = torch.stack(images, 0)

        lengths = [len(c) for c in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i,c in enumerate(captions):
            end = lengths[i]
            targets[i,:end] = c[:end]
        # Lengths indicates the valid length of each caption
        return images, targets, lengths

    crop_size = (224,224)
    trans = transforms.Compose([ 
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
    trainset = dset.CocoCaptions(root=TRAIN_IMAGE_PATH, annFile=CAPTION_PATH, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    return trainloader
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from pycocotools.coco import COCO
# import os
import pickle
from PIL import Image
# from tqdm import tqdm
from nltk import tokenize
import re
# import vocab
import random

# def tokenize_caption(sentences):
# 	cdir = os.path.dirname(os.path.abspath(__file__))+'/'

# 	# change all characters to lowercase, remove periods and extract sentences shorter than 14 words
# 	ret = []
# 	for caption in sentences:
# 		caption = re.sub(r'\'d'," had", caption)
# 		caption = re.sub(r'\'m'," am", caption)
# 		caption = re.sub(r'\'s'," is", caption)
# 		caption = re.sub(r'[&]+'," and ", caption)
# 		caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)
# 		tokens = tokenize.word_tokenize(caption.lower())
# 		if tokens[-1] == '.':
# 			tokens = tokens[:-1]
# 		# if len(tokens) <= 13:
# 		ret.append(tokens)	
	
# 	# add start token before and end token after the sentence and add padding until length of each sentences is 15
# 	with open(cdir+'../vocab/word_to_id.pkl', 'rb') as f:
# 		word_to_id = pickle.load(f)
	
# 	for i in range(len(ret)):
# 		sent = ['<start>'] + ret[i] + ['<end>']
# 		sent_id = [word_to_id[t] if (t in word_to_id) else word_to_id['<unk>'] for t in sent]
# 		# if (len(sent_id)) < 15:
# 		# 	sent_id = sent_id + [word_to_id['<pad>']] * (15-len(sent_id))
# 		ret[i] = sent_id
	
# 	return ret	
# class ():

#     def __init__(self, BATCH_SIZE,WORD_TO_ID,CAPTION_PATH,TRAIN_IMAGE_PATH):
#         self.BATCH_SIZE = BATCH_SIZE
#         self.WORD_TO_ID = WORD_TO_ID
#         self.CAPTION_PATH = CAPTION_PATH
#         self.TRAIN_IMAGE_PATH = TRAIN_IMAGE_PATH

def COCO_loader(BATCH_SIZE,WORD_TO_ID,CAPTION_PATH,TRAIN_IMAGE_PATH):

    def tokenize_caption(caption):
        # change all characters to lowercase, remove periods and extract sentences shorter than 14 words
        caption = re.sub(r'\'d'," had", caption)
        caption = re.sub(r'\'m'," am", caption)
        caption = re.sub(r'\'s'," is", caption)
        caption = re.sub(r'[&]+'," and ", caption)
        caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)
        tokens = tokenize.word_tokenize(caption.lower())
        if tokens[-1] == '.':
            tokens = tokens[:-1]


        sent = ['<start>'] + tokens + ['<end>']
        sent_id = [WORD_TO_ID[t] if (t in WORD_TO_ID) else WORD_TO_ID['<unk>'] for t in sent]
        return torch.Tensor(sent_id)

    # def load_dataset(root, annFile, transform):
    #     coco = COCO(annFile)
    #     ann_keys = coco.anns.keys()

    #     data = []
    #     with tqdm(ann_keys) as pbar:
    #         pbar.set_description("[Data Loading]")
    #         for i,key in enumerate(pbar):
    #             if i > 10241:
    #                 break
    #             caption = coco.anns[key]['caption']
    #             img_id = coco.anns[key]['image_id']
    #             path = coco.loadImgs(img_id)[0]['file_name']

    #             image = Image.open(os.path.join(root, path)).convert('RGB')
    #             if transform is not None:
    #                 image = transform(image)

    #             # Convert caption (string) to word ids.
    #             caption = tokenize_caption([caption])
    #             target = torch.Tensor(caption[0])
    #             data += [(image, target)]
    #     return data

    def collate_fn(data):
        # add start token before and end token after the sentence and add padding until length of each sentences is 15
        # with open(cdir+'../vocab/word_to_id.pkl', 'rb') as f:
        #     word_to_id = pickle.load(f)

        images, captions = zip(*data)
        # data = [(images[i].clone(), tokenize_caption(c)) for i,cap in enumerate(captions) for c in cap]
        captions = [tokenize_caption(c[random.randrange(len(c))]) for c in captions]
        # captions = [tokenize_caption(c[0]) for c in captions]
        data = zip(images, captions)
        data = sorted(data, key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # add 1D to make 4D tensor
        images = torch.stack(images, 0)

        lengths = [len(c) for c in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i,c in enumerate(captions):
            end = lengths[i]
            targets[i,:end] = c[:end]
        return images, targets, lengths
    # def collate_fn(data):
    #     data.sort(key=lambda x: len(x[1]), reverse=True)
    #     images, captions = zip(*data)
        
    #     # add 1D to make 4D tensor
    #     images = torch.stack(images, 0)

    #     lengths = [len(c) for c in captions]
    #     targets = torch.zeros(len(captions), max(lengths)).long()
    #     for i,c in enumerate(captions):
    #         end = lengths[i]
    #         targets[i,:end] = c[:end]
    #     return images, targets, lengths

    # with open(VOCAB_PATH, 'rb') as f:
    #     word_to_id = pickle.load(f)

    crop_size = (224,224)
    trans = transforms.Compose([ 
            # transforms.RandomCrop(crop_size),
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])

    # trans = transforms.Compose([transforms.Resize(img_size),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5,), (0.5,))])
        
    trainset = dset.CocoCaptions(root=TRAIN_IMAGE_PATH,
                                annFile=CAPTION_PATH,
                                transform=trans)
    # trainset = load_dataset(cdir+'../data/train/images', cdir+'../data/train/captions_train2014.json', trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, collate_fn = collate_fn)

    return trainloader
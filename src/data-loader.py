import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as dset
# from torch.nn.utils.rnn import pack_padded_sequence
from pycocotools.coco import COCO
import os
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm
from nltk import tokenize
import re
import vocab

class EncoderCNN(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, EMBEDDING_DIM)
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers, batch_first=True)
        self.linear = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, HIDDEN_DIM)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, VOCAB_SIZE)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, EMBEDDING_DIM)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, EMBEDDING_DIM)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

def tokenize_caption(sentences):
	cdir = os.path.dirname(os.path.abspath(__file__))+'/'

	# change all characters to lowercase, remove periods and extract sentences shorter than 14 words
	ret = []
	for caption in sentences:
		caption = re.sub(r'\'d'," had", caption)
		caption = re.sub(r'\'m'," am", caption)
		caption = re.sub(r'\'s'," is", caption)
		caption = re.sub(r'[&]+'," and ", caption)
		caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)
		tokens = tokenize.word_tokenize(caption.lower())
		if tokens[-1] == '.':
			tokens = tokens[:-1]
		# if len(tokens) <= 13:
		ret.append(tokens)	
	
	# add start token before and end token after the sentence and add padding until length of each sentences is 15
	with open(cdir+'../vocab/word_to_id.pkl', 'rb') as f:
		word_to_id = pickle.load(f)
	
	for i in range(len(ret)):
		sent = ['<start>'] + ret[i] + ['<end>']
		sent_id = [word_to_id[t] if (t in word_to_id) else word_to_id['<unk>'] for t in sent]
		# if (len(sent_id)) < 15:
		# 	sent_id = sent_id + [word_to_id['<pad>']] * (15-len(sent_id))
		ret[i] = sent_id
	
	return ret	

def load_dataset(root, annFile, transform):
    coco = COCO(annFile)
    ann_keys = coco.anns.keys()

    data = []
    for key in ann_keys:
        caption = coco.anns[key]['caption']
        img_id = coco.anns[key]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if transform is not None:
            image = transform(image)

        # Convert caption (string) to word ids.
        caption = tokenize_caption([caption])
        target = torch.Tensor(caption[0])
        data += [(image, target)]
    return data

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # add 1D to make 4D tensor
    images = torch.stack(images, 0)

    lengths = [len(c) for c in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i,c in enumerate(captions):
        end = lengths[i]
        targets[i,:end] = c[:end]
    return images, targets, lengths


cdir = os.path.dirname(os.path.abspath(__file__))+'/'
with open(cdir+'../vocab/word_to_id.pkl', 'rb') as f:
    word_to_id = pickle.load(f)
with open(cdir+'../vocab/filename_token.pkl', 'rb') as f:
    filename_token = pickle.load(f)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running in "+dev+".")
device = torch.device(dev)
LEARNING_RATE = 0.01
BATCH_SIZE = 2048
NUM_EPOCHS = 5
VOCAB_SIZE = len(word_to_id)+1
EMBEDDING_DIM = 10
HIDDEN_DIM = 128
# embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
# lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

# encoder = models.resnet50(pretrained=True)
# encoder.fc = nn.Identity()
encoder = EncoderCNN(EMBEDDING_DIM).to(device)
decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, 1).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

img_size = (224,224)
trans = transforms.Compose([transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])
# trainset = dset.CocoCaptions(root = cdir+'../data/train/images',
#                              annFile = cdir+'../data/train/captions_train2014.json',
#                              transform=trans)
trainset = load_dataset(cdir+'../data/train/images', cdir+'../data/train/captions_train2014.json', trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, collate_fn = collate_fn)

print("Start Training.")
# Train the models
total_step = len(trainloader)
for epoch in range(NUM_EPOCHS):
    with tqdm(trainloader) as pbar:
        pbar.set_description("[Epoch %d]" % (epoch + 1))
        for i, (images, captions,length) in enumerate(pbar):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

        # Print log info
        print('Epoch [{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, loss.item(), np.exp(loss.item()))) 
            
        # Save the model checkpoints
        model_path = cdir+'model/encoder-'+'-'+str(epoch+1)+'.pth'
        torch.save(encoder.to('cpu').state_dict(), model_path)
        encoder.to(device)
        model_path = cdir+'model/decoder-'+'-'+str(epoch+1)+'.pth'
        torch.save(decoder.to('cpu').state_dict(), model_path)
        decoder.to(device)
        # torch.save(decoder.state_dict(), os.path.join(
        #     args.model_path, cdir+' model/decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        # torch.save(encoder.state_dict(), os.path.join(
        #     args.model_path, cdir+'model/encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
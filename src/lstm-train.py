import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as dset
# from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle
import numpy as np
from tqdm import tqdm
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
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
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
VOCAB_SIZE = len(word_to_id)
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
trainset = dset.CocoCaptions(root = cdir+'../data/train/images',
                             annFile = cdir+'../data/train/captions_train2014.json',
                             transform=trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

# Train the models
total_step = len(trainloader)
for epoch in range(NUM_EPOCHS):
    with tqdm(trainloader) as pbar:
        pbar.set_description("[Epoch %d]" % (epoch + 1))
        for i, (images, captions) in enumerate(pbar):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = vocab.tokenize_caption(captions).to(device)
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            print(features)
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
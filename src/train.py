import datetime
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import EncoderCNN
from model import DecoderRNN
from data_loader import COCO_load
import os
import pickle
import numpy as np
from tqdm import tqdm

def date_print(str):
    print('[{:%Y-%m-%d %H:%M:%S}]: {}'.format(datetime.datetime.now(), str))

cdir = os.path.dirname(os.path.abspath(__file__))+'/'

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
date_print("Running in "+dev+".")
device = torch.device(dev)
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 5
# VOCAB_SIZE = len(word_to_id)+1
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1

date_print("Loading Data.")
VOCAB_SIZE, trainloader = COCO_load(BATCH_SIZE)
# embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
# lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

# encoder = models.resnet50(pretrained=True)
# encoder.fc = nn.Identity()
encoder = EncoderCNN(EMBEDDING_DIM).to(device)
decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LAYERS).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

date_print("Start Training.")
# Train the models
total_step = len(trainloader)
for epoch in range(NUM_EPOCHS):
    with tqdm(trainloader) as pbar:
        pbar.set_description("[Epoch %d]" % (epoch + 1))
        for i, (images, captions,lengths) in enumerate(pbar):
            
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
        date_print('Epoch {}, Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch+1, loss.item(), np.exp(loss.item()))) 
            
        # Save the model checkpoints
        model_path = '{}model/encoder-{}-{}-{}-{}-{}.pth'.format(cdir,epoch+1,EMBEDDING_DIM,HIDDEN_DIM,VOCAB_SIZE,NUM_LAYERS)
        torch.save(encoder.to('cpu').state_dict(), model_path)
        encoder.to(device)
        model_path = '{}model/decoder-{}-{}-{}-{}-{}.pth'.format(cdir,epoch+1,EMBEDDING_DIM,HIDDEN_DIM,VOCAB_SIZE,NUM_LAYERS)
        torch.save(decoder.to('cpu').state_dict(), model_path)
        decoder.to(device)

        # torch.save(decoder.state_dict(), os.path.join(
        #     args.model_path, cdir+' model/decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        # torch.save(encoder.state_dict(), os.path.join(
        #     args.model_path, cdir+'model/encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
date_print("Training finished")
import datetime
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .model import EncoderCNN
from .model import DecoderRNN
from .data_loader import COCO_loader
import os
import pickle
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from config import Config

def date_print(str):
    print('[{:%Y-%m-%d %H:%M:%S}]: {}'.format(datetime.datetime.now(), str))

def train():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    date_print("Running in "+dev+".")
    device = torch.device(dev)

    config = Config()
    config.train()

    LEARNING_RATE = config.LEARNING_RATE
    BATCH_SIZE = config.BATCH_SIZE
    NUM_EPOCHS = config.NUM_EPOCHS
    EMBEDDING_DIM = config.EMBEDDING_DIM
    HIDDEN_DIM = config.HIDDEN_DIM
    VOCAB_SIZE = config.VOCAB_SIZE
    NUM_LAYERS = config.NUM_LAYERS
    WORD_TO_ID = config.WORD_TO_ID

    CAPTION_PATH = config.CAPTION_PATH
    TRAIN_IMAGE_PATH = config.TRAIN_IMAGE_PATH
    MODEL_PATH = config.MODEL_PATH

    date_print("Loading Data.")
    # trainloader = COCO_loader(BATCH_SIZE,WORD_TO_ID,CAPTION_PATH,TRAIN_IMAGE_PATH)
    trainloader = COCO_loader(BATCH_SIZE,WORD_TO_ID,CAPTION_PATH,TRAIN_IMAGE_PATH)
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
                
            # Save the model
            ENCODER_FILE = '{}/encoder-{}-{}-{}-{}-{}-{:.0f}.pth'.format(MODEL_PATH,epoch+1,EMBEDDING_DIM,HIDDEN_DIM,VOCAB_SIZE,NUM_LAYERS,loss.item()*100)
            torch.save(encoder.to('cpu').state_dict(), ENCODER_FILE)
            encoder.to(device)

            DECODER_FILE = '{}/decoder-{}-{}-{}-{}-{}-{:.0f}.pth'.format(MODEL_PATH,epoch+1,EMBEDDING_DIM,HIDDEN_DIM,VOCAB_SIZE,NUM_LAYERS,loss.item()*100)
            torch.save(decoder.to('cpu').state_dict(), DECODER_FILE)
            decoder.to(device)

    date_print("Training finished")
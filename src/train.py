import datetime
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .model import EncoderCNN
from .model import DecoderRNN
from .data_loader import COCO_loader
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from config import Config

def date_print(str):
    print('[{:%Y-%m-%d %H:%M:%S}]: {}'.format(datetime.datetime.now(), str))

def train():
    # Choose Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    date_print("Running in %s." % device)

    # Import (hyper)parameters
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

    print("[Parameters]: ")
    print('\tLEARNING_RATE: {}\n\tBATCH_SIZE: {}\n\tNUM_EPOCHS: {}\n\tEMBEDDING_DIM: {}\n\tHIDDEN_DIM: {}\n\tVOCAB_SIZE: {}\n\tNUM_LAYERS: {}'.format(LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LAYERS))

    TRAIN_CAPTION_PATH = config.TRAIN_CAPTION_PATH
    TRAIN_IMAGE_PATH = config.TRAIN_IMAGE_PATH
    MODEL_PATH = config.MODEL_PATH

    date_print("Loading Data.")
    trainloader = COCO_loader(BATCH_SIZE,WORD_TO_ID,TRAIN_CAPTION_PATH,TRAIN_IMAGE_PATH)
    # Build models
    encoder = EncoderCNN(EMBEDDING_DIM).to(device)
    decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LAYERS).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    # Train the models
    date_print("Start Training.")
    for epoch in range(NUM_EPOCHS):
        with tqdm(trainloader) as pbar:
            pbar.set_description("[Epoch %d]" % (epoch + 1))
            for i, (images, captions,lengths) in enumerate(pbar):
                # Set mini-batch dataset
                images, captions = images.to(device), captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                optimizer.zero_grad()

                # Forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Print log info
            date_print('Epoch {}, Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch+1, loss.item(), np.exp(loss.item())))

            # Save the model
            ENCODER_FILE = '{}/encoder-{:02d}-{}-{:.0f}.pth'.format(MODEL_PATH,epoch+1,NUM_LAYERS,loss.item()*100)
            torch.save(encoder.to('cpu').state_dict(), ENCODER_FILE)
            encoder.to(device)

            DECODER_FILE = '{}/decoder-{:02d}-{}-{:.0f}.pth'.format(MODEL_PATH,epoch+1,NUM_LAYERS,loss.item()*100)
            torch.save(decoder.to('cpu').state_dict(), DECODER_FILE)
            decoder.to(device)

    date_print("Training finished")
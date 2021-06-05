import datetime
import torch
import torchvision.transforms as transforms
from .model import EncoderCNN
from .model import DecoderRNN
from PIL import Image
import pickle
import glob
import sys
import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from config import Config

def load_image(image_file, transform=None):
    image = Image.open(image_file)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def infer():
    # Choose Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running in %s." % device)

    # Import (hyper)parameters
    config = Config()
    config.infer()

    EMBEDDING_DIM = config.EMBEDDING_DIM
    HIDDEN_DIM = config.HIDDEN_DIM
    NUM_LAYERS = config.NUM_LAYERS
    VOCAB_SIZE = config.VOCAB_SIZE
    BEAM_SIZE = config.BEAM_SIZE
    MAX_SEG_LENGTH = config.MAX_SEG_LENGTH

    ID_TO_WORD = config.ID_TO_WORD
    END_ID = config.END_ID

    ENCODER_PATH = config.ENCODER_PATH
    DECODER_PATH = config.DECODER_PATH
    INFER_IMAGE_PATH = config.INFER_IMAGE_PATH
    INFER_RESULT_PATH = config.INFER_RESULT_PATH

    # Write log info
    with open(INFER_RESULT_PATH, 'a') as f:
        print("", file=f)
        print('---------------------', file=f)
        print('date: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), file=f)
        print('encoder model: {}'.format(ENCODER_PATH), file=f)
        print('decoder model: {}'.format(DECODER_PATH), file=f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Build models
    encoder = EncoderCNN(EMBEDDING_DIM)
    encoder = encoder.to(device).eval()

    decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LAYERS, MAX_SEG_LENGTH)
    decoder = decoder.to(device).eval()

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    decoder.load_state_dict(torch.load(DECODER_PATH))

    for image_file in sorted(glob.glob(os.path.join(INFER_IMAGE_PATH, "*"))):
        with open(INFER_RESULT_PATH, 'a') as f:
            print("", file=f)
            print("file name: {}".format(os.path.basename(image_file)), file=f)

        print("file name: {}".format(os.path.basename(image_file)))

        # Prepare an image
        image = load_image(image_file, transform).to(device)

        # Generate an caption from the image
        with torch.no_grad():
            feature = encoder(image)
            sampled_ids = decoder.beam_search(feature, BEAM_SIZE, END_ID)

        # Convert word_ids to words
        for i, (sampled_id, prob) in enumerate(sampled_ids):
            sampled_id = sampled_id.cpu().numpy()
            sampled_caption = []
            for word_id in sampled_id:
                word = ID_TO_WORD[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            print ("  {}.log p = {:.3f} '{}'".format(i+1, prob.item(), sentence))

            with open(INFER_RESULT_PATH, 'a') as f:
                print("  {}.log p = {:.3f} '{}'".format(i+1, prob.item(), sentence), file=f)
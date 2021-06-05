import datetime
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchtext.data.metrics import bleu_score
from .model import EncoderCNN
from .model import DecoderRNN
from tqdm import tqdm
from .vocab import process_sentence
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from config import Config

def date_print(str):
    print('[{:%Y-%m-%d %H:%M:%S}]: {}'.format(datetime.datetime.now(), str))

def compare():
    # Choose Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    date_print("Running in %s." % device)

    # Import (hyper)parameters
    config = Config()
    config.compare()

    EMBEDDING_DIM = config.EMBEDDING_DIM
    HIDDEN_DIM = config.HIDDEN_DIM
    NUM_LAYERS_LIST = config.NUM_LAYERS_LIST
    VOCAB_SIZE = config.VOCAB_SIZE
    BEAM_SIZE_LIST = config.BEAM_SIZE_LIST
    MAX_SEG_LENGTH = config.MAX_SEG_LENGTH
    NUM_COMPARE_IMAGES = config.NUM_COMPARE_IMAGES

    ID_TO_WORD = config.ID_TO_WORD
    END_ID = config.END_ID

    TEST_CAPTION_PATH = config.TEST_CAPTION_PATH
    TEST_IMAGE_PATH = config.TEST_IMAGE_PATH
    COMPARE_IMAGE_PATH = config.COMPARE_IMAGE_PATH
    ENCODER_PATH_LIST = config.ENCODER_PATH_LIST
    DECODER_PATH_LIST = config.DECODER_PATH_LIST
    COMPARE_RESULT_PATH = config.COMPARE_RESULT_PATH

    date_print("Loading Data.")
    crop_size = (224,224)
    trans = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    testset = dset.CocoCaptions(root=TEST_IMAGE_PATH, annFile=TEST_CAPTION_PATH)

    # Build models
    encoder = [EncoderCNN(EMBEDDING_DIM) for l in NUM_LAYERS_LIST]
    encoder = [e.to(device).eval() for e in encoder]

    decoder = [DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, l, MAX_SEG_LENGTH) for l in NUM_LAYERS_LIST]
    decoder = [d.to(device).eval() for d in decoder]

    # Load the trained model parameters
    for i in range(len(NUM_LAYERS_LIST)):
        encoder[i].load_state_dict(torch.load(ENCODER_PATH_LIST[i]))
        decoder[i].load_state_dict(torch.load(DECODER_PATH_LIST[i]))

    # Evaluate the models
    candidates = []
    references = []
    date_print("Start Comparing.")
    for i, (image, labels) in enumerate(testset):
        if i == NUM_COMPARE_IMAGES:
            break
        plt.imsave(os.path.join(COMPARE_IMAGE_PATH, '{}.jpg'.format(i)), np.array(image))

        ret = ["---------------------", "image: {}.jpg".format(i)]
        label = ' '.join(process_sentence(labels[0]))
        ret.append("ground truth: {}".format(label))
        image = torch.unsqueeze(trans(image), 0).to(device)
        with tqdm(NUM_LAYERS_LIST) as pbar:
            pbar.set_description("[Image {}]".format(i))
            for j, l in enumerate(pbar):
                for _, b in enumerate(BEAM_SIZE_LIST):
                    feature = encoder[j](image)
                    feature = feature.to(device)
                    sampled_ids = decoder[j].beam_search(feature, b, END_ID)

                    # Convert word_ids to words (only most probable one)
                    sampled_id, prob = sampled_ids[0]
                    sampled_caption = []
                    for word_id in sampled_id.cpu().numpy():
                        word = ID_TO_WORD[word_id]
                        if word == '<end>':
                            break
                        if word != '<start>':
                            sampled_caption.append(word)
                    ret.append("(num of layer, beam size) = ({:2d}, {:2d}): log p = {:.3f}, {}".format(l, b, prob, ' '.join(sampled_caption)))

        with open(COMPARE_RESULT_PATH, 'a') as f:
            for r in ret:
                print(r, file=f)
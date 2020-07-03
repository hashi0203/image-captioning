import datetime
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchtext.data.metrics import bleu_score
from .model import EncoderCNN
from .model import DecoderRNN
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from vocab.vocab import process_sentence
from config import Config

def date_print(str):
    print('[{:%Y-%m-%d %H:%M:%S}]: {}'.format(datetime.datetime.now(), str))

def evalate():
    # Choose Device
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    date_print("Running in "+dev+".")
    device = torch.device(dev)

    # Import (hyper)parameters
    config = Config()
    config.eval()

    EMBEDDING_DIM = config.EMBEDDING_DIM
    HIDDEN_DIM = config.HIDDEN_DIM
    NUM_LAYERS = config.NUM_LAYERS
    VOCAB_SIZE = config.VOCAB_SIZE
    BEAM_SIZE = config.BEAM_SIZE
    MAX_SEG_LENGTH = config.MAX_SEG_LENGTH

    ID_TO_WORD = config.ID_TO_WORD
    END_ID = config.END_ID

    TEST_CAPTION_PATH = config.TEST_CAPTION_PATH
    TEST_IMAGE_PATH = config.TEST_IMAGE_PATH
    ENCODER_PATH = config.ENCODER_PATH
    DECODER_PATH = config.DECODER_PATH
    TEST_RESULT_PATH = config.TEST_RESULT_PATH

    date_print("Loading Data.")
    crop_size = (224,224)
    trans = transforms.Compose([ 
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
    testset = dset.CocoCaptions(root=TEST_IMAGE_PATH, annFile=TEST_CAPTION_PATH, transform=trans)

    # Write log info
    with open(TEST_RESULT_PATH, 'a') as f:
        print("", file=f)
        print('---------------------', file=f)
        print('date: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), file=f)
        print('encoder model: {}'.format(ENCODER_PATH), file=f)
        print('decoder model: {}'.format(DECODER_PATH), file=f)

    # Build models
    decoder_device = torch.device('cpu')
    encoder = EncoderCNN(EMBEDDING_DIM).eval()
    decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LAYERS, MAX_SEG_LENGTH)
    encoder = encoder.to(device)
    decoder = decoder.to(decoder_device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    decoder.load_state_dict(torch.load(DECODER_PATH))

    # Evaluate the models
    candidates = []
    references = []
    date_print("Start Evaluating.")
    with tqdm(testset) as pbar:
        pbar.set_description("[Evaluating Models]")
        for i, (image, labels) in enumerate(pbar):
            # Generate an caption from the image
            image = torch.unsqueeze(image, 0).to(device)
            feature = encoder(image)
            feature = feature.to(decoder_device)
            sampled_ids = decoder.beam_search(feature, BEAM_SIZE, END_ID, decoder_device, progress=False)

            # Convert word_ids to words (only most probable one)
            sampled_id = sampled_ids[0][0].cpu().numpy()
            sampled_caption = []
            for word_id in sampled_id:
                word = ID_TO_WORD[word_id]
                if word == '<end>':
                    break
                if word != '<start>':
                    sampled_caption.append(word)
            candidates.append(sampled_caption)
            labels = [process_sentence(l) for l in labels]
            references.append(labels)
            
            if i % 1000 == 0:
                score = bleu_score(candidates, references)
                with open(TEST_RESULT_PATH, 'a') as f:
                    print("{}.bleu score = {:.3f}".format(i+1, score), file=f)

    score = bleu_score(candidates, references)
    with open(TEST_RESULT_PATH, 'a') as f:
        print("{}.bleu score = {:.3f}".format(len(testset), score), file=f)
    print(score)
        
if __name__ == "__main__":
    eval()
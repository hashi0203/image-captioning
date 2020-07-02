import os
import pickle

class Config(object):
    # Settings of (hyper)parameters
    # Set the relative path from the directory of this file
    def __init__(self):
        self.PREPARE_VOCAB = False

        self.EMBEDDING_DIM = 256
        self.HIDDEN_DIM = 512
        self.NUM_LAYERS = 1

        self.CAPTION_PATH = 'data/train/captions_train2014.json'
        self.WORD_TO_ID_PATH = 'vocab/word_to_id.pkl'
        self.ID_TO_WORD_PATH = 'vocab/id_to_word.pkl'

        # Change relative path to absolute path
        self.CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.CAPTION_PATH)
        self.WORD_TO_ID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.WORD_TO_ID_PATH)
        self.ID_TO_WORD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ID_TO_WORD_PATH)

    def train(self):
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 128
        self.NUM_EPOCHS = 20

        self.TRAIN_IMAGE_PATH = 'data/train/images'
        self.MODEL_PATH = 'model'

        # Change relative path to absolute path
        self.TRAIN_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TRAIN_IMAGE_PATH)
        self.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.MODEL_PATH)

        with open(self.WORD_TO_ID_PATH, 'rb') as f:
            self.WORD_TO_ID = pickle.load(f)
        self.VOCAB_SIZE = len(self.WORD_TO_ID)+1

        if not(os.path.isdir(self.MODEL_PATH)):
            os.makedirs(self.MODEL_PATH)
        
    def infer(self):
        self.BEAM_SIZE = 5
        self.ENCODER_PATH = 'model/encoder-20-256-512-11305-1-208.pth'
        self.DECODER_PATH = 'model/decoder-20-256-512-11305-1-208.pth'

        self.INFER_IMAGE_PATH = 'test/images/*'
        self.INFER_RESULT_PATH = 'test/result.txt'

        # Change relative path to absolute path
        self.ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ENCODER_PATH)
        self.DECODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.DECODER_PATH)
        self.INFER_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.INFER_IMAGE_PATH)
        self.INFER_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.INFER_RESULT_PATH)

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)+1

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]



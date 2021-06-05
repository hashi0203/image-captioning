import os
import pickle

class Config(object):
    # Set the relative path from the directory of this file
    # Settings of general (hyper)parameters
    def __init__(self):
        self.PREPARE_VOCAB = False

        self.EMBEDDING_DIM = 256
        self.HIDDEN_DIM = 512
        self.NUM_LAYERS = 2

        self.TRAIN_CAPTION_PATH = 'data/train/captions_train2014.json'
        self.WORD_TO_ID_PATH = 'vocab/word_to_id.pkl'
        self.ID_TO_WORD_PATH = 'vocab/id_to_word.pkl'

        # Change relative path to absolute path
        self.TRAIN_CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TRAIN_CAPTION_PATH)
        self.WORD_TO_ID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.WORD_TO_ID_PATH)
        self.ID_TO_WORD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ID_TO_WORD_PATH)

    # Settings of (hyper)parameters for training
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
        self.VOCAB_SIZE = len(self.WORD_TO_ID)

        if not(os.path.isdir(self.MODEL_PATH)):
            os.makedirs(self.MODEL_PATH)

    # Settings of (hyper)parameters for evaluation
    def eval(self):
        self.BEAM_SIZE = 5
        self.MAX_SEG_LENGTH = 20
        self.LOG_STEP = 1000
        self.NUM_EVAL_IMAGES = 5000

        self.TEST_CAPTION_PATH = 'data/val/captions_val2014.json'
        self.TEST_IMAGE_PATH = 'data/val/images'
        self.TEST_RESULT_PATH = 'log/test_results.txt'

        # # slurm-507562.out
        # self.NUM_LAYERS = 1
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-1-203.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-1-203.pth'

        # # slurm-507567.out
        self.NUM_LAYERS = 2
        self.ENCODER_PATH = 'model/encoder-20-256-512-11312-2-188.pth'
        self.DECODER_PATH = 'model/decoder-20-256-512-11312-2-188.pth'
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-2-192.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-2-192.pth'

        # # slurm-507568.out
        # self.NUM_LAYERS = 3
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-3-211.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-3-211.pth'

        # # slurm-507601.out
        # self.NUM_LAYERS = 4
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-4-229.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-4-229.pth'

        # # slurm-507613.out
        # self.NUM_LAYERS = 5
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-5-254.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-5-254.pth'

        # # slurm-507614.out
        # self.NUM_LAYERS = 6
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-6-231.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-6-231.pth'

        # Change relative path to absolute path
        self.TEST_CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_CAPTION_PATH)
        self.TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_IMAGE_PATH)
        self.TEST_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_RESULT_PATH)
        self.ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ENCODER_PATH)
        self.DECODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.DECODER_PATH)

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]

    # Settings of (hyper)parameters for inference
    def infer(self):
        self.BEAM_SIZE = 5
        self.MAX_SEG_LENGTH=20

        # # slurm-507562.out
        # self.NUM_LAYERS = 1
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-1-203.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-1-203.pth'

        # # slurm-507567.out
        self.NUM_LAYERS = 2
        self.ENCODER_PATH = 'model/encoder-20-256-512-11312-2-188.pth'
        self.DECODER_PATH = 'model/decoder-20-256-512-11312-2-188.pth'
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-2-192.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-2-192.pth'

        # slurm-507568.out
        # self.NUM_LAYERS = 3
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-3-211.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-3-211.pth'

        # # slurm-507601.out
        # self.NUM_LAYERS = 4
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-4-229.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-4-229.pth'

        # # slurm-507613.out
        # self.NUM_LAYERS = 5
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-5-254.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-5-254.pth'

        # # slurm-507614.out
        # self.NUM_LAYERS = 6
        # self.ENCODER_PATH = 'model/encoder-20-256-512-11312-6-231.pth'
        # self.DECODER_PATH = 'model/decoder-20-256-512-11312-6-231.pth'

        self.INFER_IMAGE_PATH = 'images'
        self.INFER_RESULT_PATH = 'log/infer_results.txt'

        # Change relative path to absolute path
        self.ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ENCODER_PATH)
        self.DECODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.DECODER_PATH)
        self.INFER_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.INFER_IMAGE_PATH)
        self.INFER_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.INFER_RESULT_PATH)

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]

    # Settings of (hyper)parameters for comparison
    def compare(self):
        self.BEAM_SIZE_LIST = [1, 5, 10] # Set beam size that you want to try
        self.MAX_SEG_LENGTH = 20
        self.NUM_COMPARE_IMAGES = 10

        self.TEST_CAPTION_PATH = 'data/val/captions_val2014.json'
        self.TEST_IMAGE_PATH = 'data/val/images'
        self.COMPARE_IMAGE_PATH = 'log/compare_images'
        self.COMPARE_RESULT_PATH = 'log/compare_results.txt'

        # Set encoder and decoder model pathes to compare
        self.ENCODER_PATH_LIST = ['model/encoder-10-256-512-11312-2-215.pth',
                                  'model/encoder-20-256-512-11312-2-188.pth']
        self.DECODER_PATH_LIST = ['model/decoder-10-256-512-11312-2-215.pth',
                                  'model/decoder-20-256-512-11312-2-188.pth']

        # Set the number of layers of the models to compare
        self.NUM_LAYERS_LIST = [2, 2]

        # Change relative path to absolute path
        self.TEST_CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_CAPTION_PATH)
        self.TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_IMAGE_PATH)
        self.COMPARE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.COMPARE_IMAGE_PATH)
        self.COMPARE_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.COMPARE_RESULT_PATH)
        self.ENCODER_PATH_LIST = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in self.ENCODER_PATH_LIST]
        self.DECODER_PATH_LIST = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in self.DECODER_PATH_LIST]

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]
import torch
import torchvision.transforms as transforms
from model import EncoderCNN
from model import DecoderRNN
from PIL import Image
import os
import pickle
import glob

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

cdir = os.path.dirname(os.path.abspath(__file__))+'/'
with open(cdir+'../vocab/id_to_word.pkl', 'rb') as f:
    id_to_word = pickle.load(f)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running in "+dev+".")
device = torch.device(dev)
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 5
VOCAB_SIZE = len(id_to_word)+1
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
BEAM_SIZE = 5

encoder_path = cdir + 'model/encoder-5-256-512-11305-1.pth'
decoder_path = cdir + 'model/decoder-5-256-512-11305-1.pth'
image_dir = cdir + '../test/images/*'

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                        (0.229, 0.224, 0.225))])

# Build models
encoder = EncoderCNN(EMBEDDING_DIM).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_LAYERS)
encoder = encoder.to(device)
decoder = decoder.to(device)

# Load the trained model parameters
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

for image_path in glob.glob(image_dir):
    # Prepare an image
    print("file name: {}".format(os.path.basename(image_path)))

    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature, BEAM_SIZE)

    # Convert word_ids to words
    for i, (sampled_id, prob) in enumerate(sampled_ids):
        sampled_id = sampled_id.cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        sampled_caption = []
        for word_id in sampled_id:
            word = id_to_word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        print (" {}.p = {:.3f} '{}'".format(i, prob.item(), sentence))

# Print out the image and the generated caption
# image = Image.open(image_path)
# plt.imshow(np.asarray(image))
# plt.show()
# import matplotlib.pyplot as plt
import argparse
from config import Config
from vocab import vocab
from src.train import train
from src.eval import evalate
from src.infer import infer

# def save_img(img_name, img):
# 	img_np = img.to('cpu').detach().numpy().copy()
# 	plt.imsave(img_name, img_np[0])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
            prog='main.py',
            description='This program enables you to make and test image captioning model.',
            epilog='end',
            add_help=True,
            )
	
	parser.add_argument('phase', help='to designate the phase (train, eval, infer)')
	phase = parser.parse_args().phase

	config = Config()
	
	if config.PREPARE_VOCAB:
		vocab.prepare_vocab(config.TRAIN_CAPTION_PATH, config.WORD_TO_ID_PATH, config.ID_TO_WORD_PATH)

	if phase == 'train':
		train()
	elif phase == 'eval':
		evalate()
	elif phase == 'infer':
		infer()
	else:
		print('the argument should be train, eval, or infer')
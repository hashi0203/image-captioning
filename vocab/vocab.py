from nltk import tokenize
from pycocotools.coco import COCO
from collections import Counter
import pickle
import re
from tqdm import tqdm

def process_sentence(caption):
	# Transform abbreviation
	caption = re.sub(r'\'d'," had", caption, flags=re.IGNORECASE)
	caption = re.sub(r'\'m'," am", caption, flags=re.IGNORECASE)
	caption = re.sub(r'let\'s'," lets", caption, flags=re.IGNORECASE)
	caption = re.sub(r'\'s'," is", caption, flags=re.IGNORECASE)
	caption = re.sub(r'\'re'," are", caption, flags=re.IGNORECASE)
	caption = re.sub(r'[@]+'," at ", caption)
	caption = re.sub(r'[&]+'," and ", caption)
	# Remove special characters
	caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]\\/\"]+'," ", caption)
	tokens = tokenize.word_tokenize(caption.lower())
	return tokens


def prepare_vocab(TRAIN_CAPTION_PATH, WORD_TO_ID_PATH, ID_TO_WORD_PATH):
	# Read caption info
	coco = COCO(TRAIN_CAPTION_PATH)
	anns_keys = coco.anns.keys()
	
	original_token = []
	# longest_token_length = 0
	with tqdm(anns_keys) as pbar:
		pbar.set_description("[Preparing Vocab]")
		for key in pbar:
			caption = coco.anns[key]['caption']
			original_token.extend(process_sentence(caption))

	# Count the appearance of each words
	freq = Counter(original_token)
	
	# Extract words which appear more than twice
	common = freq.most_common()
	vocab = sorted([t for t,c in common if c>=3])
	# Add start, end, and unknown tokens
	vocab.append('<start>')
	vocab.append('<end>')
	vocab.append('<unk>')
	
	# Save vocabs
	word_to_id = {t:i for i,t in enumerate(vocab)}
	with open(WORD_TO_ID_PATH, 'wb') as f:
		pickle.dump(word_to_id, f)
	
	id_to_word = {i:t for i,t in enumerate(vocab)}
	with open(ID_TO_WORD_PATH, 'wb') as f:
		pickle.dump(id_to_word, f)
	
	print('num of vocabs: ' + str(len(word_to_id)))
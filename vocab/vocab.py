from nltk import tokenize
from pycocotools.coco import COCO
from collections import Counter
import pickle
import re
from tqdm import tqdm


def prepare_vocab(CAPTION_PATH, WORD_TO_ID_PATH, ID_TO_WORD_PATH):
	# Read caption info
	coco = COCO(CAPTION_PATH)
	anns_keys = coco.anns.keys()
	
	original_token = []
	longest_token_length = 0
	with tqdm(anns_keys) as pbar:
		pbar.set_description("[Preparing Vocab]")
		for key in pbar:
			caption = coco.anns[key]['caption']
			# Transform abbreviation
			caption = re.sub(r'\'d'," had", caption)
			caption = re.sub(r'\'m'," am", caption)
			caption = re.sub(r'\'s'," is", caption)
			caption = re.sub(r'[&]+'," and ", caption)
			# Remove special characters
			caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)

			tokens = tokenize.word_tokenize(caption.lower())
			if len(tokens) > longest_token_length:
				longest_token_length = len(tokens)
			img_id = coco.anns[key]['image_id']
			tmp = [coco.loadImgs(img_id)[0]['file_name'], tokens]
			original_token.append(tmp)

	# Count the appearance of each words
	freq = Counter()
	for i in range(len(original_token)):
		freq.update(set(original_token[i][1]))
	
	# Extract words which appear more than twice
	common = freq.most_common()
	vocab = sorted([t for t,c in common if c>=3])
	# Add start, end, and unknown tokens
	vocab.append('<start>')
	vocab.append('<end>')
	vocab.append('<unk>')
	
	# Save vocabs
	word_to_id = {t:i+1 for i,t in enumerate(vocab)}
	with open(WORD_TO_ID_PATH, 'wb') as f:
		pickle.dump(word_to_id, f)
	
	id_to_word = {i+1:t for i,t in enumerate(vocab)}
	with open(ID_TO_WORD_PATH, 'wb') as f:
		pickle.dump(id_to_word, f)
	
	print('num of vocabs: ' + str(len(word_to_id)))
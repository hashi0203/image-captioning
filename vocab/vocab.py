from nltk import tokenize
from pycocotools.coco import COCO
from collections import Counter
import pickle
# import os
import re
from tqdm import tqdm


def prepare_vocab(CAPTION_PATH, WORD_TO_ID_PATH, ID_TO_WORD_PATH):
	# caption_file = cdir + '../data/train/captions_train2014.json'
	# vocab_files = ['id_to_word.pkl', 'word_to_id.pkl', 'filename_token.pkl', 'id_to_image.pkl', 'image_to_id.pkl', 'imageid_token.pkl']
	# vocab_files = [cdir + f for f in vocab_files] 
	# # do nothing if vocab files exist and are newer
	# flag = 0
	# for f in vocab_files:
	# 	if not(os.path.isfile(f) and os.path.getmtime(caption_file) < os.path.getmtime(f) and os.path.getmtime(__file__) < os.path.getmtime(f)):
	# 		flag = 1
	# 		break

	# if flag == 0:
	# 	print('vocab files are not changed')
	# 	return 
	# if opt:
	# 	print('vocab files are not changed')
	# 	return 
	
	# read caption info
	coco = COCO(CAPTION_PATH)
	# print('num of original train images: ' + str(len(coco.imgs)))
	anns_keys = coco.anns.keys()
	
	# change all characters to lowercase, remove periods and extract sentences shorter than 14 words
	original_token = []
	
	longest_token_length = 0
	with tqdm(anns_keys) as pbar:
		pbar.set_description("[Preparing Vocab]")
		for key in pbar:
			caption = coco.anns[key]['caption']
			caption = re.sub(r'\'d'," had", caption)
			caption = re.sub(r'\'m'," am", caption)
			caption = re.sub(r'\'s'," is", caption)
			caption = re.sub(r'[&]+'," and ", caption)
			caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)

			tokens = tokenize.word_tokenize(caption.lower())
			if len(tokens) > longest_token_length:
				longest_token_length = len(tokens)
			img_id = coco.anns[key]['image_id']
			tmp = [coco.loadImgs(img_id)[0]['file_name'], tokens]
			original_token.append(tmp)
		
	# print("longest token lenght: "+str(longest_token_length+2))

	# count the appearance of each words
	freq = Counter()
	for i in range(len(original_token)):
		freq.update(set(original_token[i][1]))
	
	# extract words which appear more than twice
	common = freq.most_common()
	vocab = sorted([t for t,c in common if c>=3])
	# add start, end, unknown, and padding tokens
	vocab.append('<start>')
	vocab.append('<end>')
	vocab.append('<unk>')
	# vocab.append('<pad>')
	
	# add start token before and end token after the sentence and add padding until length of each sentences is 15
	word_to_id = {t:i+1 for i,t in enumerate(vocab)}
	id_to_word = {i+1:t for i,t in enumerate(vocab)}
	
	# original_token_id = [] 
	
	# for i in range(len(original_token)):
	# 	sent = ['<start>'] + original_token[i][1] + ['<end>']
	# 	sent_id = [word_to_id[t] if (t in word_to_id) else word_to_id['<unk>'] for t in sent]
	# 	# if (len(sent_id)) < 15:
	# 	# 	sent_id = sent_id + [word_to_id['<pad>']] * (15-len(sent_id))
	# 	original_token_id.append([original_token[i][0], sent_id])
	
	# save vocabs
	with open(WORD_TO_ID_PATH, 'wb') as f:
		pickle.dump(word_to_id, f)
	
	with open(ID_TO_WORD_PATH, 'wb') as f:
		pickle.dump(id_to_word, f)
	
	# with open(vocab_files[2], 'wb') as f:
	# 	pickle.dump(original_token_id, f)
	
	# print('num of train images: ' + str(len(list(set([c[0] for c in original_token_id])))))
	print('num of vocabs: ' + str(len(word_to_id)))
	
	# # change file extention from jpg to png
	# png_token = []
	
	# for i in range(len(original_token_id)):
	# 	name, ext = os.path.splitext(original_token_id[i][0])
	# 	new_filename = name+'.png'
	# 	png_token.append([new_filename, original_token_id[i][1]])
	
	# # add ids to filename
	# tmp_list = list(set([c[0] for c in png_token]))
	# id_to_image = {i+1:t for i,t in enumerate(tmp_list)}
	# image_to_id = {t:i+1 for i,t in enumerate(tmp_list)}
	
	# imageid_token = [[image_to_id[c[0]], c[1]] for c in png_token]
	
	# # save vocabs
	# with open(vocab_files[3], 'wb') as f:
	# 	pickle.dump(id_to_image,f)
	
	# with open(vocab_files[4], 'wb') as f:
	# 	pickle.dump(image_to_id, f)
	
	# with open(vocab_files[5], 'wb') as f:
	# 	pickle.dump(imageid_token, f)

# def tokenize_caption(sentences):
# 	cdir = os.path.dirname(os.path.abspath(__file__))+'/'

# 	# change all characters to lowercase, remove periods and extract sentences shorter than 14 words
# 	ret = []
# 	for caption in sentences[0]:
# 		caption = re.sub(r'\'d'," had", caption)
# 		caption = re.sub(r'\'m'," am", caption)
# 		caption = re.sub(r'\'s'," is", caption)
# 		caption = re.sub(r'[&]+'," and ", caption)
# 		caption = re.sub(r'[!.,:;#$>\'\`\?\-\(\)\[\]]+'," ", caption)
# 		tokens = tokenize.word_tokenize(caption.lower())
# 		if tokens[-1] == '.':
# 			tokens = tokens[:-1]
# 		# if len(tokens) <= 13:
# 		ret.append(tokens)	
	
# 	# add start token before and end token after the sentence and add padding until length of each sentences is 15
# 	with open(cdir+'id_to_word.pkl', 'rb') as f:
# 		id_to_word = pickle.load(f)

# 	with open(cdir+'word_to_id.pkl', 'rb') as f:
# 		word_to_id = pickle.load(f)
	
# 	for i in range(len(ret)):
# 		sent = ['<start>'] + ret[i] + ['<end>']
# 		sent_id = [word_to_id[t] if (t in word_to_id) else word_to_id['<unk>'] for t in sent]
# 		# if (len(sent_id)) < 15:
# 		# 	sent_id = sent_id + [word_to_id['<pad>']] * (15-len(sent_id))
# 		ret[i] = sent_id
	
	# return ret	

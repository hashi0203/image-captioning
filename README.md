# Image Captioning

This application is to train, evaluate and infer image captioning.
Image captioning takes images as input and return the caption of the images as output.
This program is based on ["Show and Tell: A Neural Image Caption Generator" by Vinayls et al. (ICML2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf), and implemented using "Pytorch".

# DEMO

|<img src="https://github.com/hashi0203/image-captioning/blob/master/test/images/cat.jpg?raw=true" alt="img0" width="400px"> |<img src="https://github.com/hashi0203/image-captioning/blob/master/test/images/1.jpg?raw=true" alt="img1" width="400px">                     |
|:-------------------------------:|:-----------------------------:|
|a brown and white cat sitting in the grass    |a red double decker bus driving down a street |

|<img src="https://github.com/hashi0203/image-captioning/blob/master/test/images/2.jpg?raw=true" alt="img2" width="400px"> |<img src="https://github.com/hashi0203/image-captioning/blob/master/test/images/3.jpg?raw=true" alt="img3" width="400px">                     |
|:-------------------------------:|:-----------------------------:|
|a group of people sitting around a table eating food            |a man riding a wave on top of a surfboard            |

# Requirement

 - Python==3.6.3
 - nltk==3.5
 - numpy==1.19.0
 - Pillow==7.1.2
 - pycocotools==2.0
 - torch==1.5.1+cu101
 - torchvision==0.6.1+cu101
 - tqdm==4.46.1
 - torchtext==0.6.0

# Installation

1. Paste the following commands at a terminal prompt to download this source code.

	```bash
	git clone https://github.com/hashi0203/image-captioning.git
	```
2. There are some options to construct the environment.
	 - pipenv
		```bash
		cd image-captioning
		pipenv install
		```
	 - pip
		```bash
		cd image-captioning
		pip install -r requirements.txt
		```

# Usage

 - Train
	 1. Download datasets and captions.  
		This application uses MSCOCO dataset, and you can download them from following links.  
		The train dataset is for training, the validation dataset is for evaluation, so you can download only what you need.
 
		- [2014 Train images [83K/13GB]](http://images.cocodataset.org/zips/train2014.zip)  
		- [2014 Val images [41K/6GB]](http://images.cocodataset.org/zips/val2014.zip)
		- [2014 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

	2. Place downloaded files under the 'data' directory following the directory tree.
		```bash
		|-- image-captioning
		    |-- data
			    |-- train
				    |-- captions_train2014.json
				    |-- images
					    |-- COCO_train2014_{number}.jpg
					    |-- ..
				|-- val
					|-- captions_val2014.json
				    |-- images
					    |-- COCO_val2014_{number}.jpg
					    |-- ..
		    |-- ..
		``` 

	 3. Set (hyper)parameters in config.py.
		It is also ok if you don't edit anything.

	4. Start training by using the Encoder CNN to change images to feature vectors and the Decoder RNN (LSTM) to change feature vectors to captions.
		```bash
		python3 main.py 'train'
		```
		Ref. It took 6.5 hours to complete with the default parameters by 4 GPUs in NVIDIA TESLA P100(Pascal).  
		Ref. Final loss was 1.9155 in this case.

	5. Model files are save in image-captioning/model if you didn't edit config.py.

- Evaluate
	1. Following 1, 2, 3 in Training section.
	2. Set (hyper)parameters in config.py.
		It is also ok if you don't edit anything, but be sure that the parameters should be the same as when training.
	3. Start evaluating using the BLEU-4 score.
		```bash
		python3 main.py 'eval'
		```
		Ref. It took very long time if you use all images, so I recommend you to save outputs by setting LOG_STEP and stop evaluating when the values are stable. 
		Ref. The Decoder RNN model runs faster if you use it in CPU than GPU.
		Ref. BLEU-4 score by 5000 images was 0.266 in this case.
	4. Output will be shown in stdout and also you can check it in image-captioning/test/test_results.txt.

- Infer
	1. Prepare images which you want to make captions of and place it in image-captioning/test/images.
	2. Set (hyper)parameters in config.py.
		It is also ok if you don't edit anything, but be sure that the parameters should be the same as when training.
	3. Start inferring by beam search.
		```bash
		python3 main.py 'infer'
		```
	4. Output will be shown in stdout and also you can check it in image-captioning/test/infer_results.txt.

# Reference
```bash
@article{DBLP:journals/corr/VinyalsTBE14,
  author    = {Oriol Vinyals and
               Alexander Toshev and
               Samy Bengio and
               Dumitru Erhan},
  title     = {Show and Tell: {A} Neural Image Caption Generator},
  journal   = {CoRR},
  volume    = {abs/1411.4555},
  year      = {2014},
  url       = {http://arxiv.org/abs/1411.4555},
  archivePrefix = {arXiv},
  eprint    = {1411.4555},
  timestamp = {Mon, 13 Aug 2018 16:47:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/VinyalsTBE14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
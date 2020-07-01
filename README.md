# Image Captioning

This application is to train, validate and infer image captioning.
Image captioning takes images as input and return the caption of the images as output.
This program is based on "Show and Tell: A Neural Image Caption Generator" by Vinayls et al. (ICML2015), and implemented using "Pytorch".

# DEMO

### Todo

# Requirement

 - Python==3.6.3
 - nltk==3.5
 - numpy==1.19.0
 - Pillow==7.1.2
 - pycocotools==2.0
 - torch==1.5.1+cu101
 - torchvision==0.6.1+cu101
 - tqdm==4.46.1

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

	4. Start training.
		```bash
		python3 main.py 'train'
		```

	5. Model files are save in image-captioning/model if you didn't edit config.py.

- Validate
	1. Following 1, 2, 3 in Training section.
	2. Todo

- Infer
	1. Prepare images which you want to make captions of and place it in image-captioning/test/images.
	2. Set (hyper)parameters in config.py.
		It is also ok if you don't edit anything, but be sure that the parameters should be the same as when training.
	3. Start inferring.
		```bash
		python3 main.py 'infer'
		```
	4. Output will be shown in stdout and also you can check it in image-captioning/test/result.txt.

# coding: utf-8
# from data_provider import *
from C3D_model import *
import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os 
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import skimage.io as io
from skimage.transform import resize
import h5py
from PIL import Image

MODEL_DIRECTORY_PATH = os.path.join(os.path.dirname(__file__), "models")
PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIRECTORY_PATH,"c3d.pickle")

DATASET_DRECTORY_PATH = os.path.join(os.path.dirname(__file__), "datasets")
UTE_VIDEO_PATH = os.path.join(DATASET_DRECTORY_PATH,"UTE_Video")

gen_video_path = DATASET_DRECTORY_PATH + "/UTE_Video/"
gen_output_h5_path = DATASET_DRECTORY_PATH + "/UTE_Datatset_Features/C3D_features/"

OUTPUT_DIR = gen_output_h5_path
EXTRACTED_LAYER = [6]
VIDEO_DIR = gen_video_path
RUN_GPU = True
OUTPUT_NAME = "c3d_features_new.hdf5"
BATCH_SIZE = 128
crop_w = 112
resize_w = 128
crop_h = 112
resize_h = 171
nb_frames = 16
name = ["P01.mp4", "P02.mp4", "P03.mp4", "P04.mp4"]


def feature_extractor():
	#trainloader = Train_Data_Loader( VIDEO_DIR, resize_w=128, resize_h=171, crop_w = 112, crop_h = 112, nb_frames=16)
	net = C3D(487)
	print('net', net)
	## Loading pretrained model from sports and finetune the last layer
	net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
	if RUN_GPU : 
		net.cuda(0)
		net.eval()
		print('net', net)
	# feature_dim = 4096 if EXTRACTED_LAYER != 5 else 8192

	# read video list from the folder 
	#video_list = os.listdir(VIDEO_DIR)

	# read video list from the txt list
	# video_list_file = name
	video_list = name
	video_list = [item.strip() for item in video_list]
	print('video_list', video_list)

	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	f = h5py.File(os.path.join(OUTPUT_DIR, OUTPUT_NAME), 'w')

	# current location
	temp_path = os.path.join(os.getcwd(), 'temp')
	if not os.path.exists(temp_path):
		os.mkdir(temp_path)

	error_fid = open('error.txt', 'w')
	for video_name in video_list: 
		video_path = os.path.join(VIDEO_DIR, video_name)
		print('video_path', video_path)
		frame_path = os.path.join(temp_path, video_name)
		if not os.path.exists(frame_path):
			os.mkdir(frame_path)


		print('Extracting video frames ...')
		# using ffmpeg to extract video frames into a temporary folder
		# example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
		os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%5d.jpg')


		print('Extracting features ...')
		total_frames = len(os.listdir(frame_path))
		if total_frames == 0:
			error_fid.write(video_name+'\n')
			print('Fail to extract frames for video: %s'%video_name)
			continue

		valid_frames = total_frames // nb_frames * nb_frames
		n_feat = int(valid_frames // nb_frames)
		n_batch = int(n_feat // BATCH_SIZE) 
		if n_feat - n_batch*BATCH_SIZE > 0:
			n_batch = n_batch + 1
		print('n_frames: %d; n_feat: %d; n_batch: %d'%(total_frames, n_feat, n_batch))
		
		#print 'Total frames: %d'%total_frames 
		#print 'Total validated frames: %d'%valid_frames
		#print 'NB features: %d' %(valid_frames/nb_frames)
		index_w = np.random.randint(resize_w - crop_w) ## crop
		index_h = np.random.randint(resize_h - crop_h) ## crop

		features = []

		for i in range(n_batch-1):
			input_blobs = []
			for j in range(BATCH_SIZE):
				clip = []
				clip = np.array([resize(io.imread(os.path.join(frame_path, 'image_{:05d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
				#print('clip_shape', clip.shape)
				clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
				#print('clip_shape',clip.shape)
				#print('range', range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1)))	
				input_blobs.append(clip)
			input_blobs = np.array(input_blobs, dtype=np.float32)
			print('input_blobs_shape', input_blobs.shape)
			input_blobs = torch.from_numpy(np.array(input_blobs.transpose(0, 4, 1, 2, 3), dtype=np.float32))
			input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
			_, batch_output = net(input_blobs, 5)	
			batch_feature  = (batch_output.data).cpu()
			features.append(batch_feature)

		# The last batch
		input_blobs = []
		for j in range(n_feat-(n_batch-1)*BATCH_SIZE):
			clip = []
			clip = np.array([resize(io.imread(os.path.join(frame_path, 'image_{:05d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range(((n_batch-1)*BATCH_SIZE+j) * nb_frames+1, min(((n_batch-1)*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])

			clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
			#print('range', range(((n_batch-1)*BATCH_SIZE+j) * nb_frames+1, min(((n_batch-1)*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1)))
			input_blobs.append(clip)
		input_blobs = np.array(input_blobs, dtype='float32')
		#print('input_blobs_shape', input_blobs.shape)
		input_blobs = torch.from_numpy(np.array(input_blobs.transpose(0, 4, 1, 2, 3), dtype=np.float32))
		input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
		_, batch_output = net(input_blobs, 5)
		batch_feature  = (batch_output.data).cpu()
		features.append(batch_feature)

		features = torch.cat(features, 0)
		features = features.numpy()
		print('features', features)
		fgroup = f.create_group(video_name)
		fgroup.create_dataset('c3d_features', data=features)
		fgroup.create_dataset('total_frames', data=np.array(total_frames))
		fgroup.create_dataset('valid_frames', data=np.array(valid_frames))

		print ('%s has been processed...'%video_name)


		# clear temp frame folders
		try: 
			os.system('rm -rf ' + frame_path)
		except: 
			pass

if __name__ == '__main__':
    feature_extractor()

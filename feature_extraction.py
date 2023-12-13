import cv2
import numpy as np
import h5py
import os
from googlenet_pytorch import GoogLeNet 
import json
import torch
import torchvision.transforms as transforms
from PIL import Image


model = GoogLeNet.from_pretrained('googlenet')
model.eval()
model.fc = torch.nn.Identity()
model.eval()
print(model)
# import pdb;pdb.set_trace()
preprocess = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# "Y:\rl\project\interact_summ_code\datasets\UTE_Video\P01.mp4"

DATASET_DRECTORY_PATH = os.path.join(os.path.dirname(__file__), "datasets")
UTE_VIDEO_PATH = os.path.join(DATASET_DRECTORY_PATH,"UTE_Video")
# print(DATASET_DRECTORY_PATH, UTE_VIDEO_PATH)

def extract_cnn_features(video_path, target_fps, output_file):
    # Open the video file
    # print(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames to achieve the target_fps
    interval = int(round(fps / target_fps))
    # import pdb;pdb.set_trace()
    features = []

    frame_count, num_added = 0,0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Check if the current frame should be processed based on the interval
        
        if frame_count % interval == 0:
            # Preprocess the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            # print()
            
            # Extract features using the model
            with torch.no_grad():
                cnn_features = model(input_tensor)
            # cnn_features = cnn_features.reshape(-1, 1024)
            cnn_features = cnn_features.flatten()
            # print(input_tensor.shape, cnn_features.shape)
            features.append(cnn_features.cpu().detach().numpy())
            num_added+=1

        frame_count += 1
        if frame_count%1000==0:
            print(frame_count, num_added)
            print(input_tensor.shape, cnn_features.shape)

    # Release the video capture object
    cap.release()
    # features = features.cpu().detach().numpy()
    features = np.array(features)

    print(f"Features extracted from the video at {target_fps}fps.")
    print(f"Shape of the feature matrix: {features.shape}")

    # Save features to H5PY file
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('features', data=features)

    print(f"Features saved to {output_file}")

    return features

# Example usage
# Y:\rl\project\interact_summ_code\datasets\UTE_Video\P01.mp4
# video_path = UTE_VIDEO_PATH + "/P01.mp4"
gen_video_path = DATASET_DRECTORY_PATH + "/UTE_Video/"
gen_output_h5_path = DATASET_DRECTORY_PATH + "/UTE_Datatset_Features/CNN _features/"
target_fps = 2.14


name = ["P01", "P02", "P03", "P04"]
for n in name:
    video_path = gen_video_path + f"{n}.mp4"
    output_h5_path = gen_output_h5_path + f"{n}_cnn_features.h5"
    print(video_path)
    print(output_h5_path)
    video_features = extract_cnn_features(video_path, target_fps, output_h5_path)


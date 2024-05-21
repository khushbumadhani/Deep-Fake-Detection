from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from flask import Flask, request, jsonify, render_template
import time
import os

app = Flask(__name__)

model = models.inception_v3(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 10),
    nn.Linear(10, 2)
)
model.load_state_dict(torch.load('./model.pth',map_location=torch.device('cpu')))
model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),  # Resize the image to match the model's input size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

def extract_faces(video_path):
  cap = cv2.VideoCapture(video_path)
  mtcnn = MTCNN(select_largest=False, post_process=False)
  # Get video name without extension
  video_name = os.path.splitext(os.path.basename(video_path))[0]

  # Create folder for frames if it doesn't exist
  output_dir = f'./data//{video_name}_frames'
  os.makedirs(output_dir, exist_ok=True)
  # Detect face
  frame_count=0
  face_list = []
  while cap.isOpened():
    frame_count+=1
    ret, frame = cap.read()
    if not ret:
      break
    face = mtcnn(frame)
    if face is None:
      continue
    face_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(face_path, face.permute(1, 2, 0).int().numpy())
    face_list.append(face)
  cap.release()
  return face_list

'''
def API():
  faces = extract_faces("/content/drive/MyDrive/CyberDefence AI/Dataset/dfdc_train_part_0/htorvhbcae.mp4")
  input_images = [transform(face) for face in faces]
  # Stack the list of tensors into a single tensor
  input_images = torch.stack(input_images)

  # Perform inference with the model
  with torch.no_grad():
    predictions = model(input_images)
  
  pred = predictions.numpy()
  fake_predictions = pred[:, 0]  # Index 0 corresponds to fake predictions
  real_predictions = pred[:, 1]  # Index 1 corresponds to real predictions
  average_fake = np.mean(fake_predictions)
  average_real = np.mean(real_predictions)
  if real_predictions > fake_predictions:
    return "REAL"
  else:
    return "FAKE"
'''

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'video_file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    video_file = request.files['video_file']
    # Timestamp for unique filename
    timestamp = str(int(time.time()))
    video_path = './videos/' + timestamp + '_' + video_file.filename
    video_file.save(video_path)
    
    # Face extraction
    print('Face Extraction Started')
    faces = extract_faces(video_path)
    input_images = [transform(face) for face in faces]
    print('Face Extraction Ended')
    
    # Prediction
    print('Prediction Started')
    input_images = torch.stack(input_images)
    with torch.no_grad():
      predictions = model(input_images)
    print('Prediction Ended')
    
    pred = predictions.numpy()
    fake_predictions = pred[:, 0]  # Index 0 corresponds to fake predictions
    real_predictions = pred[:, 1]  # Index 1 corresponds to real predictions
    average_fake = np.mean(fake_predictions)
    average_real = np.mean(real_predictions)

    if average_real > average_fake:
        result = {'prediction': 'REAL'}
    else:
        result = {'prediction': 'FAKE'}

    return jsonify(result)
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


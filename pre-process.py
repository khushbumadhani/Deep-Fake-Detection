import os
import glob
import cv2
from tqdm import tqdm
import pandas as pd
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
import numpy as np

def extract_faces(video_path):
  cap = cv2.VideoCapture(video_path)
  mtcnn = MTCNN(select_largest=False, post_process=False)
  # Get video name without extension
  video_name = os.path.splitext(os.path.basename(video_path))[0]

  # Create folder for frames if it doesn't exist
  output_dir = f'./data/{video_name}_frames'
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

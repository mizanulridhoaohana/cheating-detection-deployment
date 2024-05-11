# Library 

from roboflow import Roboflow
from ultralytics import YOLO

# Load Model
model = YOLO('./model/best.pt')

video_path = "./video/testing_video.mp4"
model.predict(video_path, save=True, imgsz=640, conf=0.35, line_width=2, save_txt=True)
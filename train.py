import torch
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # Load the pretrained model

# Specify your training data configuration in a YAML file
data_yaml = r"C:\Users\Safi.uddin\TASK\Coco\safi.yaml"



# Train the model
results = model.train(
    data=data_yaml,  # Path to your data configuration YAML file
    epochs=10,       # Number of training epochs
    imgsz=640        # Image size used for training
)
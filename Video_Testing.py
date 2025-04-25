'''
To test the YOLOv8 model on the video data run this code snippet. 
Here I have a trained YOLO model loaded in my environment and then there isa video 
available in project directory that is selected.
This model is just trained on a small set of data which has hands wiht rings only.
Model only know about one class that is a ring, so as expected this does not perform
well on the Video data, due to the lack of diverese data.
'''


# Import YOLO model
from ultralytics import YOLO

# Here I already have a trained best model parameters 
model = YOLO("Models/best.pt")  

# select a video for inference frame by frame and save in a runs directory
results = model("Video_Files/219228_small.mp4", save=True)

# The output is stored in a directory
print("Detection complete. Check 'runs/detect/predict'")
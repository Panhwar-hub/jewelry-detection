Anna's Jewelry Detection Assistant

Hey! This is a simple little project I’ve been working on that’s all about detecting rings on hands in images. I trained this on a small custom dataset — around 3,179 images — where hands with rings are labeled. The idea is to combine a few powerful tools to get a clear view of where the rings are, both via segmentation and object detection.

Here’s how it works under the hood:
I’m using UNet to segment the ring areas from the hand — this gives a nice mask showing where the ring might be.
Then, YOLOv8 (a super fast object detection model) picks up the ring as an object and draws bounding boxes.
For finger and hand tracking, I’m using MediaPipe, which makes it really easy to visualize hand landmarks.
All of this comes together in a Python simple app built with Tkinter, so you can load an image, see the results, and play around with it locally.

Features
Hand Tracking – using MediaPipe to show landmarks and finger joints.
Ring Segmentation – via UNet model trained on a small dataset.
Ring Detection – YOLOv8 model detects ring locations with boxes.
User Interface – select your image and see all 3 results side by side.

git clone https://github.com/yourusername/Jewelry-Detection-Assistant.git
cd Jewelry-Detection-Assistant

2. Set up a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

3. Install the dependencies
pip install -r requirements.txt

4. Download the trained UNet and YOLOv8 models:

- UNet: https://drive.google.com/file/d/1wtjdfbj7xDH-g78xJcBjnzkBTR7ZMqQI/view?usp=sharing
- Place it in the `Models/` folder as `jewelry_unet.pth`
- Place your YOLOv8 `best.pt` model in the same folder

5. Put your models in the right place
Your jewelry_unet.pth into the models/ folder
Your YOLOv8 .pt file (best.pt) into the same folder

Start the GUI
python app/app.py
It’ll open up a window. Click “Select an Image,” and it will show:

The original image with detected hands
A segmentation map of where the ring is
The YOLO detection box showing ring location


Tech Stack
Python (3.10+)
Tkinter (for the GUI)
PyTorch
YOLOv8 (from Ultralytics)
MediaPipe
OpenCV + PIL for image processing

Improvements I'm Thinking Of
Webcam input for live detection
Ring classification by type/material
Option to save the output results
Better dataset to improve segmentation accuracy

Final Thoughts
This is something I put together for fun and learning, so it’s far from perfect — but I think it’s a neat combination of tools and works surprisingly well on most clear hand images. Feel free to try it out, use your own models, or tweak it however you like!

'''
Following block of code makes the function calls to run the inference and provide results 
from the YOLO, UNet and MediaPipe models. 
'''
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import mediapipe as mp
import torch

# I have used here mediapipe pretrained models to track the fingers
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# YOLO ring detection
def run_yolo_detection(model, image_path):
    results = model(image_path)[0].plot()
    return Image.fromarray(results[..., ::-1])

# UNet ring segmentation
def run_unet_segmentation(model, image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    mask = output.squeeze().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, image.size)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mask_rgb)

# Track the fingers using Mediapipe
def run_hand_tracking(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return Image.fromarray(img_rgb)

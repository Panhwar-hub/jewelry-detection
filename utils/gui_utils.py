'''
To visualize the model results I have used Tkinter python library to provide a 
simple GUI with model inferences. This version has ability to upload an image from 
local directory and gives reults for hand tracking, ring segmentation, and ring detection. 
'''


import tkinter as tk
from PIL import ImageTk

# Function expects three images Yolo detection, segmentation mask, and hand trcking. 
def display_results(result_frame, yolo_img, unet_img, hand_img):
    for widget in result_frame.winfo_children():
        widget.destroy()


    results = [yolo_img, unet_img, hand_img]
    labels = ["Finger Tracking", "UNet Segmentation", "YOLOv8 Detection"]

    # Visualize these reults. 
    for img, lbl in zip(results, labels):
        img_tk = ImageTk.PhotoImage(img)
        panel = tk.Label(result_frame, image=img_tk, text=lbl, compound="top", font=("Arial", 12), bg="#f0f0f5")
        panel.image = img_tk
        panel.pack(side="left", padx=10, pady=10)

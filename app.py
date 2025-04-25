'''
APP.py
Is the main python file that brings all the pieces of code in action, calling this python 
file would run a GUI to load an image and provide with three different resulting images. 
'''


# Required Libraries
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from utils.models import load_unet_model, load_yolo_model
from utils.processing import run_yolo_detection, run_unet_segmentation, run_hand_tracking
from utils.gui_utils import display_results

# Load UNet and YOLO models saved weights
unet_model = load_unet_model("Models/jewelry_unet.pth")
yolo_model = load_yolo_model("Models/best.pt")

# This load_image function let's you load an image from your local machine for testing
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        status_label.config(text="Processing image...", fg="blue")
        root.update_idletasks()

        # Run image through all the models
        yolo_img = run_yolo_detection(yolo_model, file_path).resize((350, 350))
        unet_img = run_unet_segmentation(unet_model, file_path).resize((350, 350))
        hand_img = run_hand_tracking(file_path).resize((350, 350))

        # Display the results, three different images. 
        display_results(result_frame, hand_img, unet_img, yolo_img)
        status_label.config(text="Process complete", fg="green")

# Here is a basic GUI based on Tkinter library 
root = tk.Tk()
root.title("Anna's Jewelry Detector") # A simple name 
root.geometry("1150x700")
root.configure(bg="#f0f0f5")

title_label = tk.Label(root, text="Anna's Jewelry Detector", font=("Helvetica", 22, "bold"), bg="#f0f0f5", fg="#4a4a7d")
title_label.pack(pady=20)

# A button to load an image from local files
select_button = tk.Button(
    root,
    text="Select an Image",
    command=load_image,
    font=("Helvetica", 14),
    bg="#4a90e2",
    fg="white",
    padx=20,
    pady=10
)
select_button.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 11), bg="#f0f0f5", fg="gray")
status_label.pack()

result_frame = tk.Frame(root, bg="#f0f0f5")
result_frame.pack(pady=20)

root.mainloop()

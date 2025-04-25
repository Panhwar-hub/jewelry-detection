
'''
I am using a Hands dataset from Kaggle, this dataset has around 11k hand images
and some of images also have accessories such as rings, bracelets or watchs. 
So I just need those images that have some accessaries, this code removes all those images 
that don't have any accessories, or if their segmentation mask does'nt exist. 
'''


import os

image_dir = "YOLO_Dataset/images"
mask_dir = "SegmentationClass"

# Create a base address for the segmented images
mask_basenames = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}

# Loop thorugh all images and remove that are not in segmentation class
for img_file in os.listdir(image_dir):
    img_basename, img_ext = os.path.splitext(img_file)
    if img_basename not in mask_basenames:
        img_path = os.path.join(image_dir, img_file)
        os.remove(img_path)
    else:
        print(f"Kept: {img_file}")

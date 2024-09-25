import os
import random
import shutil

# Define folder paths
train_images_folder = '../../../../../references/yolov5/dataset/train/images'
train_labels_folder = '../../../../../references/yolov5/dataset/train/labels'  # Corresponding labels for images  # Corresponding labels for images
val_images_folder = '../../../../../references/yolov5/dataset/val/images'
val_labels_folder = '../../../../../references/yolov5/dataset/val/labels'  

# List all training images (handling both .jpg and .JPG files)
all_images = [f for f in os.listdir(train_images_folder) if f.lower().endswith('.jpg')]

# Randomly select 10-20% of images for validation
num_val_images = int(0.2 * len(all_images))  # 20% for validation
val_images = random.sample(all_images, num_val_images)

# Move the selected images and their labels to the validation folder
for image in val_images:
    # Handle both .jpg and .JPG extensions
    image_lower = image.lower()
    image_path = os.path.join(train_images_folder, image)
    label_path = os.path.join(train_labels_folder, image_lower.replace('.jpg', '.txt'))
    
    # Ensure the file exists regardless of case
    if not os.path.exists(image_path):
        # Try with .JPG extension if .jpg not found
        image_path = os.path.join(train_images_folder, image.replace('.jpg', '.JPG'))
    
    # Move the image and its corresponding annotation file to the validation folder
    if os.path.exists(image_path) and os.path.exists(label_path):
        shutil.move(image_path, val_images_folder)
        shutil.move(label_path, val_labels_folder)

print("Validation set created successfully.")
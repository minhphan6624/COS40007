import os
import random
import shutil

# Paths to the original training dataset (images and labels in YOLO format)
train_images_dir = 'data/images/train'
train_labels_dir = 'data/converted_train_labels'  

# Paths to the destination dataset
selected_train_images_dir = 'dataset/train/images'
selected_train_labels_dir = 'dataset/train/labels'

# Make directories for selected train/test images and labels if they don't exist
os.makedirs(selected_train_images_dir, exist_ok=True)
os.makedirs(selected_train_labels_dir, exist_ok=True)

# Get all image filenames for training and testing
train_image_filenames = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.JPG'))]

# Randomly select 400 images for training and 40 for testing
selected_train_images = random.sample(train_image_filenames, 400)

# Function to copy images and corresponding labels
def copy_images_and_labels(image_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    for image in image_list:
        # Copy image
        src_image_path = os.path.join(source_img_dir, image)
        dest_image_path = os.path.join(dest_img_dir, image)
        shutil.copy(src_image_path, dest_image_path)
        
        # Copy corresponding label (.txt file)
        label_file = image.replace('.jpg', '.txt').replace('.JPG', '.txt')
        src_label_path = os.path.join(source_lbl_dir, label_file)
        if os.path.exists(src_label_path):  # Ensure label exists before copying
            dest_label_path = os.path.join(dest_lbl_dir, label_file)
            shutil.copy(src_label_path, dest_label_path)

# Copy selected training images and labels
copy_images_and_labels(selected_train_images, train_images_dir, train_labels_dir, selected_train_images_dir, selected_train_labels_dir)


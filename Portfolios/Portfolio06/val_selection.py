import os
import random
import shutil

# Define the percentage of training data to be used for validation
val_split_percentage = 0.2  # 20% for validation

# Paths to the training dataset
train_images_dir = 'dataset/train/images'
train_labels_dir = 'dataset/train/labels'

# Paths to the validation dataset
val_images_dir = 'dataset/val/images'
val_labels_dir = 'dataset/val/labels'

# Create validation directories if they don't exist
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get all training image filenames
train_image_filenames = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.JPG'))]

# Number of validation images to select
num_val_images = int(len(train_image_filenames) * val_split_percentage)

# Randomly select images for validation
val_images = random.sample(train_image_filenames, num_val_images)

# Function to move images and corresponding labels from train to val
def move_images_and_labels(image_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    for image in image_list:
        # Move image
        src_image_path = os.path.join(src_img_dir, image)
        dest_image_path = os.path.join(dest_img_dir, image)
        shutil.move(src_image_path, dest_image_path)
        
        # Move corresponding label (.txt file)
        label_file = image.replace('.jpg', '.txt').replace('.JPG', '.txt')
        src_label_path = os.path.join(src_lbl_dir, label_file)
        if os.path.exists(src_label_path):  # Ensure label exists before moving
            dest_label_path = os.path.join(dest_lbl_dir, label_file)
            shutil.move(src_label_path, dest_label_path)

# Move selected images and labels to validation set
move_images_and_labels(val_images, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)

print(f"Moved {num_val_images} images and labels to the validation set.")

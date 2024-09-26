import os
import random
import shutil

# Paths to the original test dataset 
test_images_dir = 'data/images/test'

selected_test_images_dir = 'dataset/test/images'

os.makedirs(selected_test_images_dir, exist_ok=True)

# Get all image filenames for training and testing
test_image_filenames = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.JPG'))]

# Randomly select 400 images for training and 40 for testing
selected_test_images = random.sample(test_image_filenames, 40)

# Function to copy only images (for test set)
def copy_images_only(image_list, source_img_dir, dest_img_dir):
    for image in image_list:
        src_image_path = os.path.join(source_img_dir, image)
        dest_image_path = os.path.join(dest_img_dir, image)
        shutil.copy(src_image_path, dest_image_path)

# Copy selected test images (no labels yet)
copy_images_only(selected_test_images, test_images_dir, selected_test_images_dir)

print("Selected test images have been copied.")
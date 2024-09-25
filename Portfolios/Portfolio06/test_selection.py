import os
import random
import shutil

# Define folder paths
test_images_source_folder = 'data/images/test'  # Folder where all test images are stored

# YOLOv5 test folder (where the selected test images will be moved)
yolo_test_images_folder = '../../../../../references/yolov5/dataset/test/images'

# Ensure the YOLOv5 test images folder is cleared before copying new images
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_folder(yolo_test_images_folder)

# List all test images (handling both .jpg and .JPG extensions)
all_test_images = [f for f in os.listdir(test_images_source_folder) if f.lower().endswith('.jpg')]

# Randomly select 40 test images
selected_test_images = random.sample(all_test_images, 40)

# Copy selected images to the YOLOv5 test images folder
for image in selected_test_images:
    image_path = os.path.join(test_images_source_folder, image)
    
    # Move the image to the YOLOv5 test images folder
    shutil.copy(image_path, yolo_test_images_folder)

print(f"40 test images have been successfully copied to the YOLOv5 test folder.")

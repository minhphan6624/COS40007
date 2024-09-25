import os
import random
import shutil
import pandas as pd

# Define folder paths
images_folder = 'data/images/train'
selected_images_folder = '../../../../../references/yolov5/dataset/train/images'
selected_annotations_folder = '../../../../../references/yolov5/dataset/train/labels'

# Clear the selected images and annotations folders
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_folder(selected_images_folder)
clear_folder(selected_annotations_folder)

# Load the annotations CSV file
annotations_df = pd.read_csv('data/train_yolo_annotations.csv')

# Normalize file extensions by converting them to lowercase
annotations_df['filename'] = annotations_df['filename'].str.lower()
unique_images = annotations_df['filename'].unique()

# Randomly select 400 images
selected_images = random.sample(list(unique_images), 400)

# Filter the annotations DataFrame to keep only the selected images
selected_annotations_df = annotations_df[annotations_df['filename'].isin(selected_images)]

# Copy selected images and create corresponding YOLO annotation files
for image in selected_images:
    # Handle both .jpg and .JPG cases
    image_lower = image.lower()
    image_path = os.path.join(images_folder, image_lower)
    
    # Check if the file exists with either extension
    if not os.path.exists(image_path):
        # Try with .JPG extension if .jpg not found
        image_path = os.path.join(images_folder, image_lower.replace('.jpg', '.JPG'))
    
    # Copy the image to the selected images folder
    shutil.copy(image_path, selected_images_folder)
    
    # Prepare annotation file path
    annotation_path = os.path.join(selected_annotations_folder, image_lower.replace('.jpg', '.txt'))

    # Get the annotations for this image
    image_annotations = selected_annotations_df[selected_annotations_df['filename'] == image_lower]
    
    # Write the annotation file in YOLO format
    with open(annotation_path, 'w') as f:
        for _, row in image_annotations.iterrows():
            class_id = int(row['class_id'])
            x_center = row['x_center']
            y_center = row['y_center']
            box_width = row['box_width']
            box_height = row['box_height']
            f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

print("400 images and corresponding annotations have been selected and saved.")

# import os
# import random
# import shutil
# import pandas as pd

# # Define folder paths
# images_folder = 'data/images/train'
# annotations_file = 'data/train_yolo_annotations.csv'
# selected_images_folder = '../../../../../references/yolov5/dataset/train/images'
# selected_annotations_folder = '../../../../../references/yolov5/dataset/train/labels'
# val_images_folder = '../../../../../references/yolov5/dataset/val/images'
# val_labels_folder = '../../../../../references/yolov5/dataset/val/labels'

# # Clear the selected images, annotations, and validation folders
# def clear_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path):
#             os.remove(file_path)

# clear_folder(selected_images_folder)
# clear_folder(selected_annotations_folder)
# clear_folder(val_images_folder)
# clear_folder(val_labels_folder)

# # Load the annotations CSV file
# annotations_df = pd.read_csv(annotations_file)

# # Normalize file extensions by converting them to lowercase
# annotations_df['filename'] = annotations_df['filename'].str.lower()
# unique_images = annotations_df['filename'].unique()

# # Randomly select 10-20% of images for validation
# num_val_images = int(0.2 * len(unique_images))  # 20% for validation
# val_images = random.sample(list(unique_images), num_val_images)

# # Filter out validation images from the training images
# train_images = list(set(unique_images) - set(val_images))

# # Randomly select 400 images for training
# selected_images = random.sample(train_images, 400)

# # Filter the annotations DataFrame to keep only the selected images for training
# selected_annotations_df = annotations_df[annotations_df['filename'].isin(selected_images)]

# # Copy selected training images and create corresponding YOLO annotation files
# for image in selected_images:
#     image_lower = image.lower()
#     image_path = os.path.join(images_folder, image_lower)

#     if not os.path.exists(image_path):
#         image_path = os.path.join(images_folder, image_lower.replace('.jpg', '.JPG'))
    
#     shutil.copy(image_path, selected_images_folder)
    
#     # Prepare annotation file path
#     annotation_path = os.path.join(selected_annotations_folder, image_lower.replace('.jpg', '.txt'))

#     # Get the annotations for this image
#     image_annotations = selected_annotations_df[selected_annotations_df['filename'] == image_lower]
    
#     # Write the annotation file in YOLO format
#     with open(annotation_path, 'w') as f:
#         for _, row in image_annotations.iterrows():
#             class_id = int(row['class_id'])
#             x_center = row['x_center']
#             y_center = row['y_center']
#             box_width = row['box_width']
#             box_height = row['box_height']
#             f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

# # Now move the validation images and their labels
# for image in val_images:
#     image_lower = image.lower()
#     image_path = os.path.join(images_folder, image)

#     label_path = os.path.join(selected_annotations_folder, image_lower.replace('.jpg', '.txt'))

#     if not os.path.exists(image_path):
#         image_path = os.path.join(images_folder, image.replace('.jpg', '.JPG'))

#     if os.path.exists(image_path) and os.path.exists(label_path):
#         shutil.move(image_path, val_images_folder)
#         shutil.move(label_path, val_labels_folder)

# print("400 training images and validation set created successfully.")

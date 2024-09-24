import os
import random
import shutil
import pandas as pd

# Define folder paths
images_folder = 'data/images/train'
selected_images_folder = '../../../../../references/yolov5/dataset/train/images'
selected_annotations_folder = '../../../../../references/yolov5/dataset/train/labels'

# Load the annotations CSV file
annotations_csv = 'data/train_yolo_annotations.csv'
annotations_df = pd.read_csv(annotations_csv)

# List all unique image files in the annotations
unique_images = annotations_df['filename'].unique()

# Randomly select 400 images
selected_images = random.sample(list(unique_images), 400)

# Filter the annotations DataFrame to keep only the selected images
selected_annotations_df = annotations_df[annotations_df['filename'].isin(selected_images)]

# Move selected images and create corresponding YOLO annotation files
for image in selected_images:
    image_path = os.path.join(images_folder, image)
    annotation_path = os.path.join(selected_annotations_folder, image.replace('.jpg', '.txt'))
    
    # Copy the image to the selected images folder
    shutil.copy(image_path, selected_images_folder)
    
    # Get the annotations for this image
    image_annotations = selected_annotations_df[selected_annotations_df['filename'] == image]
    
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

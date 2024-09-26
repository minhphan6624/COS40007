import os
import pandas as pd

# Load the ground truth CSV
ground_truth_csv = 'data/bounding_boxes/test_labels.csv'
output_dir = 'data/gt_boxes_yolo'  # Directory to save the .txt files

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV file into a DataFrame
df = pd.read_csv(ground_truth_csv)

# Function to convert bounding boxes to YOLO format (normalized)
def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

# Loop through each row in the CSV and generate .txt files for each image
for _, row in df.iterrows():
    image_name = row['filename']
    img_width = int(row['width'])
    img_height = int(row['height'])
    class_id = 0  # Assuming 'Graffiti' is the only class and assigned class_id 0
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

    # Convert to YOLO format (normalized)
    x_center, y_center, width, height = convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height)
    
    # Prepare the corresponding .txt file path
    # Handles both .jpg and .JPG formats
    txt_filename = os.path.join(output_dir, image_name.replace('.jpg', '.txt').replace('.JPG', '.txt'))

    # Write the bounding box in YOLO format to the .txt file
    with open(txt_filename, 'a') as f:  # 'a' to append, in case multiple boxes exist for the same image
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Conversion to YOLO format completed.")
import pandas as pd
import os

# Load the ground truth data from the test_labels.csv file
test_labels_path = 'data/bounding_boxes/test_labels.csv'  # Update this with the correct path
test_labels_df = pd.read_csv(test_labels_path)

# Folder where YOLOv5 detection results are saved
detection_folder = '../../../yolov5/runs/detect/exp2/labels'  

# Path to the folder where your test images are stored
test_images_folder = 'dataset/test/images'  # Replace with the path to your test images folder

# Dynamically get the list of 40 selected test images from the folder
selected_images = [f for f in os.listdir(test_images_folder) if f.endswith('.jpg')]

# Filter the test labels to only include the images in the test folder
test_labels_df = test_labels_df[test_labels_df['filename'].isin(selected_images)]

# Function to convert YOLOv5 predictions from txt format to bounding box format
def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Converts YOLO format to [xmin, ymin, xmax, ymax]."""
    x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    return [xmin, ymin, xmax, ymax]

# Function to compute IoU between two bounding boxes
def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection box
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Compute the area of the intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        intersection = 0

    # Compute the area of both boxes
    area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Compute IoU
    union = area_box1 + area_box2 - intersection
    iou = intersection / union if union != 0 else 0
    return iou

# Prepare output list to store all detections per image
results = []

# Iterate through ground truth and predictions
for idx, row in test_labels_df.iterrows():
    filename = row['filename']
    img_width, img_height = row['width'], row['height']
    gt_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]

    # Check if prediction exists for this image
    detection_file = os.path.join(detection_folder, filename.replace('.jpg', '.txt'))
    if os.path.exists(detection_file):
        with open(detection_file, 'r') as f:
            highest_confidence = 0
            highest_iou = 0
            for line in f:
                # Parse prediction line
                data = line.strip().split()
                class_id = int(data[0])
                x_center, y_center, width, height = map(float, data[1:5])
                confidence = float(data[5])

                # Convert YOLO format to bounding box format
                pred_box = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)

                # Compute IoU
                iou = compute_iou(gt_box, pred_box)

                # Keep track of the prediction with the highest confidence and IoU
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    highest_iou = iou

            # Store the result for the image
            results.append([filename, highest_confidence, highest_iou])
    else:
        # If no prediction exists, add default entry with confidence and IoU of 0
        results.append([filename, 0, 0])

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results, columns=['image_name', 'confidence_value', 'IoU_value'])
output_csv = 'outputCSV/iter-no.csv'  # Update the path where you want to save the file
results_df.to_csv(output_csv, index=False)

print(f"IoU results saved to {output_csv}")
